"""Document-level semantic search over semantically chunked long documents.

:func:`~sentence_transformers.util.retrieval.semantic_search` returns one hit per
embedded unit. When a corpus contains long documents that were compressed into a
single embedding, or that were chunked and embedded independently, several chunks of
the same document can crowd the top of the ranking. :class:`DocumentRetriever`
adds a document-aware layer on top of the regular encoding and scoring code:

- documents are chunked on sentence boundaries (see
  :mod:`sentence_transformers.util.chunking`),
- each chunk is encoded through the model's unchanged ``encode`` method,
- chunk scores are aggregated into a document score with a configurable strategy,
- each document appears at most once in the result, with the chunks that produced
  its score attached so the cited passage always matches the reported score.

The retriever keeps its chunk plan and chunk embeddings so that changing the query
does not re-chunk or re-encode the corpus, can persist them to disk, and can update
only the documents whose text changed.

The class is disabled by passing ``enabled=False``; in that mode it defers entirely
to ``model.encode`` plus ``semantic_search`` over one whole-document embedding per
document and returns the same hits, scores and order as before.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import Tensor

from .chunking import ChunkingConfig, ChunkingError, ChunkPlan, TextChunk, chunk_document
from .retrieval import semantic_search
from .similarity import cos_sim

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)

Aggregation = Literal["max", "mean_top_n", "hit_count_weighted"]

AGGREGATIONS: tuple[str, ...] = ("max", "mean_top_n", "hit_count_weighted")

# Format version for the on-disk index. Bumped whenever the layout changes so a stale
# cache is rebuilt instead of being misread.
INDEX_FORMAT_VERSION = 1

# Number of queries scored against the chunk matrix at once. Caps the transient
# [queries, chunks] score matrix so search memory does not grow with the corpus.
_QUERY_CHUNK_SIZE = 64


class DocumentEncodingError(RuntimeError):
    """Raised when a document cannot be chunked or encoded.

    The failing document keeps whatever chunks and embeddings it had before the call,
    so a failure never replaces a good index entry with a partial one.

    Args:
        doc_id: Identifier of the document that failed.
        chunk_index: Index of the chunk that failed, if the failure happened during encoding.
        start_char: Character offset of the failing chunk, if known.
        cause: The original exception.
    """

    def __init__(
        self,
        doc_id: str,
        chunk_index: int | None,
        start_char: int | None,
        cause: BaseException,
    ) -> None:
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.cause = cause
        location = ""
        if chunk_index is not None:
            location = f", chunk {chunk_index}"
        if start_char is not None:
            location += f" (characters [{start_char}, ...) of the document"
        super().__init__(f"Failed to encode document {doc_id!r}{location}: {cause}")
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class DocumentHit:
    """One document in a search result.

    Args:
        doc_id: Identifier of the document.
        score: Aggregated document score.
        matched_chunks: The highest-scoring chunks of this document, ordered by score.
        chunk_scores: Chunk-level similarity scores aligned with ``matched_chunks``.
        snippets: Text snippets of ``matched_chunks`` clipped to ``snippet_max_chars``.
        coverage: Fraction of the document (in characters) covered by the matched chunks
            attached to this hit, counted over the union of their character ranges so
            overlapping chunks are not double counted.
        num_chunks: Total number of indexed chunks of this document.
        hit_count: Number of chunks scoring at least ``hit_threshold``.
        aggregation: Aggregation strategy that produced ``score``.
    """

    doc_id: str
    score: float
    matched_chunks: tuple[TextChunk, ...]
    chunk_scores: tuple[float, ...]
    snippets: tuple[str, ...]
    coverage: float
    num_chunks: int
    hit_count: int
    aggregation: str

    @property
    def best_chunk(self) -> TextChunk:
        """The highest-scoring matched chunk."""
        return self.matched_chunks[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation, including source locations."""
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "aggregation": self.aggregation,
            "coverage": self.coverage,
            "num_chunks": self.num_chunks,
            "hit_count": self.hit_count,
            "matched_chunks": [
                {**chunk.to_dict(), "score": score, "snippet": snippet}
                for chunk, score, snippet in zip(self.matched_chunks, self.chunk_scores, self.snippets)
            ],
        }


@dataclass(slots=True)
class _DocumentRecord:
    # Content hash of the document text, used to detect unchanged documents.
    text_hash: str
    text: str
    plan: ChunkPlan
    embeddings: Tensor

    def __len__(self) -> int:
        return len(self.plan.chunks)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_text_map(documents: Mapping[str, str]) -> None:
    if not isinstance(documents, Mapping):
        raise TypeError(f"documents must be a mapping of doc_id to text, got {type(documents).__name__}")
    for doc_id, text in documents.items():
        if not isinstance(doc_id, str):
            raise TypeError(f"Document identifiers must be strings, got {type(doc_id).__name__}: {doc_id!r}")
        if not isinstance(text, str):
            raise TypeError(f"Document {doc_id!r} must be a string, got {type(text).__name__}")


def _whole_document_plan(doc_id: str, text: str) -> ChunkPlan:
    """The single-chunk plan used when chunking is disabled."""
    chunk = TextChunk(
        doc_id=doc_id,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        end_boundary="end",
    )
    return ChunkPlan(doc_id=doc_id, text_length=len(text), chunks=(chunk,), total_chunks=1)


def _covered_fraction(plan: ChunkPlan, hit_flags: Sequence[bool]) -> float:
    """Fraction of the document covered by the union of the hit chunk ranges."""
    ranges = sorted((chunk.start_char, chunk.end_char) for chunk, is_hit in zip(plan.chunks, hit_flags) if is_hit)
    if not ranges or not plan.text_length:
        return 0.0
    covered = 0
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return covered / plan.text_length


class DocumentRetriever:
    """Chunk long documents and rank documents, not individual chunks.

    Args:
        model: A ``SentenceTransformer`` used to encode chunks and queries. Its
            ``encode`` signature and behaviour are not modified.
        config: Chunking parameters. Defaults to ``ChunkingConfig()``.
        aggregation: How chunk scores become a document score. Defaults to
            ``"mean_top_n"`` so that a document cannot win on a single lucky chunk:

            - ``"max"``: the highest chunk score,
            - ``"mean_top_n"``: the mean of the top ``mean_top_n`` chunk scores,
            - ``"hit_count_weighted"``: ``best_score + (1 - best_score) * hit_ratio`` where
              ``hit_ratio`` is the fraction of chunks scoring at least ``hit_threshold``.

        mean_top_n: Number of top chunks used by ``"mean_top_n"``. Defaults to 3.
        hit_threshold: Minimum chunk score counted as a hit by ``"hit_count_weighted"``
            and by the coverage metric. The default of 0.0 counts every chunk with a
            non-negative cosine similarity as a hit; raise it (e.g. to 0.5) to make both
            measures reflect strongly matching chunks only.
        snippet_max_chars: Maximum length of the snippet returned per matched chunk.
        enabled: When ``False``, chunking and aggregation are bypassed entirely and
            :meth:`search` reproduces ``semantic_search`` over whole-document
            embeddings. Defaults to ``True``.
        max_chunks_per_result: Maximum number of matched chunks attached to a hit.
            Defaults to ``mean_top_n`` for the ``"mean_top_n"`` aggregation, else 1.
            For ``"mean_top_n"`` it must be at least ``mean_top_n`` so the attached
            chunks always account for the reported score.
        encode_kwargs: Extra keyword arguments forwarded to ``model.encode``, e.g.
            ``{"batch_size": 64}``. ``convert_to_tensor`` and ``convert_to_numpy``
            are always set by the retriever.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        config: ChunkingConfig | None = None,
        *,
        aggregation: Aggregation = "mean_top_n",
        mean_top_n: int = 3,
        hit_threshold: float = 0.0,
        snippet_max_chars: int = 200,
        enabled: bool = True,
        max_chunks_per_result: int | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        if not isinstance(mean_top_n, int) or mean_top_n < 1:
            raise ValueError(f"mean_top_n must be a positive integer, got {mean_top_n!r}")
        if snippet_max_chars < 0:
            raise ValueError(f"snippet_max_chars must be non-negative, got {snippet_max_chars!r}")

        self.model = model
        self.config = config or ChunkingConfig()
        self.aggregation = aggregation
        self.mean_top_n = mean_top_n
        self.hit_threshold = hit_threshold
        self.snippet_max_chars = snippet_max_chars
        self.enabled = enabled
        self.encode_kwargs = dict(encode_kwargs or {})

        if max_chunks_per_result is None:
            max_chunks_per_result = max(1, mean_top_n if aggregation == "mean_top_n" else 1)
        if max_chunks_per_result < 1:
            raise ValueError(f"max_chunks_per_result must be positive, got {max_chunks_per_result!r}")
        if aggregation == "mean_top_n" and max_chunks_per_result < mean_top_n:
            raise ValueError(
                "max_chunks_per_result must be at least mean_top_n for the 'mean_top_n' aggregation, "
                f"otherwise the matched chunks cannot account for the document score "
                f"(got {max_chunks_per_result} < {mean_top_n})"
            )
        self.max_chunks_per_result = max_chunks_per_result

        self._records: dict[str, _DocumentRecord] = {}
        # Documents that were empty or whitespace-only and were therefore not indexed.
        self._skipped: dict[str, str] = {}
        # Documents whose most recent encoding attempt failed; their old record (if any) is retained.
        self._failed: dict[str, DocumentEncodingError] = {}

    # ------------------------------------------------------------------ indexing

    @property
    def doc_ids(self) -> list[str]:
        """Identifiers of all indexed documents, in insertion order."""
        return list(self._records)

    @property
    def skipped_docs(self) -> dict[str, str]:
        """Documents that were not indexed, mapped to the reason (``"empty"``)."""
        return dict(self._skipped)

    @property
    def failed_docs(self) -> dict[str, DocumentEncodingError]:
        """Documents whose most recent encoding attempt failed."""
        return dict(self._failed)

    @property
    def num_chunks(self) -> int:
        """Total number of chunks across all indexed documents."""
        return sum(len(record) for record in self._records.values())

    def chunk_plans(self) -> dict[str, ChunkPlan]:
        """Return the chunk plan of every indexed document."""
        return {doc_id: record.plan for doc_id, record in self._records.items()}

    def _encode_texts(self, texts: Sequence[str]) -> Tensor:
        """Encode texts with the model's unchanged ``encode`` method."""
        embeddings = self.model.encode(
            list(texts),
            **{
                **self.encode_kwargs,
                "convert_to_tensor": True,
                "convert_to_numpy": False,
            },
        )
        if not isinstance(embeddings, Tensor):
            embeddings = torch.as_tensor(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        return embeddings.detach().to("cpu", dtype=torch.float32)

    def _encode_chunks(self, doc_id: str, plan: ChunkPlan) -> Tensor:
        """Encode all chunks of one document, identifying the failing chunk if any."""
        texts = [chunk.text for chunk in plan.chunks]
        try:
            return self._encode_texts(texts)
        except Exception as batch_error:
            # Retry chunk by chunk so the error names the exact chunk that failed. This
            # only runs on the failure path, so it costs nothing for healthy documents.
            for chunk in plan.chunks:
                try:
                    self._encode_texts([chunk.text])
                except Exception as chunk_error:
                    raise DocumentEncodingError(
                        doc_id, chunk.chunk_index, chunk.start_char, chunk_error
                    ) from chunk_error
            raise DocumentEncodingError(doc_id, None, None, batch_error) from batch_error

    def _plan_for(self, doc_id: str, text: str) -> ChunkPlan:
        if not self.enabled:
            return _whole_document_plan(doc_id, text)
        try:
            return chunk_document(doc_id, text, self.config)
        except ChunkingError as error:
            raise DocumentEncodingError(doc_id, None, error.position, error) from error

    def _make_record(self, doc_id: str, text: str, plan: ChunkPlan, embeddings: Tensor) -> _DocumentRecord:
        if len(embeddings) != len(plan.chunks):
            raise DocumentEncodingError(
                doc_id,
                None,
                None,
                ValueError(f"encoder returned {len(embeddings)} embeddings for {len(plan.chunks)} chunks"),
            )
        return _DocumentRecord(text_hash=_content_hash(text), text=text, plan=plan, embeddings=embeddings)

    def _encode_batch(self, plans: Sequence[tuple[str, str, ChunkPlan]]) -> dict[str, Tensor]:
        """Encode the chunks of several documents in one ``encode`` call.

        Falls back to per-document encoding only if the combined call fails, so that
        the failing document and chunk can be named while the others still succeed.
        """
        texts: list[str] = []
        slices: dict[str, slice] = {}
        for doc_id, _, plan in plans:
            start = len(texts)
            texts.extend(chunk.text for chunk in plan.chunks)
            slices[doc_id] = slice(start, len(texts))

        try:
            embeddings = self._encode_texts(texts)
            return {doc_id: embeddings[span] for doc_id, span in slices.items()}
        except Exception:
            pass

        encoded: dict[str, Tensor] = {}
        for doc_id, _, plan in plans:
            encoded[doc_id] = self._encode_chunks(doc_id, plan)
        return encoded

    def add_documents(
        self,
        documents: Mapping[str, str],
        *,
        force: bool = False,
        strict: bool = True,
    ) -> list[str]:
        """Add or update documents, re-chunking only those whose text changed.

        The update is staged and committed at the end, so a failure with ``strict=True``
        leaves the index exactly as it was before the call (no mix of old and new
        entries). A single document's failure never overwrites its previous chunks.

        Args:
            documents: Mapping of document identifier to text.
            force: Re-chunk and re-encode even documents whose text is unchanged.
                Defaults to ``False``.
            strict: Raise :class:`DocumentEncodingError` on the first failure and commit
                nothing. When ``False``, the failure is recorded, the remaining
                documents are still processed, and the successful ones are committed.
                Defaults to ``True``.

        Returns:
            list[str]: Identifiers of the documents that were actually re-encoded.

        Raises:
            DocumentEncodingError: If a document fails and ``strict=True``.
        """
        _validate_text_map(documents)

        # Plan every changed document first: chunking is cheap and deterministic, and it
        # lets all chunk texts of all changed documents share one encode call.
        plans: list[tuple[str, str, ChunkPlan]] = []
        empty: list[str] = []
        changed: list[str] = []
        failed_ids: set[str] = set()
        for doc_id, text in documents.items():
            text_hash = _content_hash(text)
            existing = self._records.get(doc_id)
            if existing is not None and not force and existing.text_hash == text_hash:
                continue
            if existing is None and not force and doc_id in self._skipped and not text.strip():
                continue
            if not text.strip() and self.enabled:
                empty.append(doc_id)
                changed.append(doc_id)
                continue
            try:
                plan = self._plan_for(doc_id, text)
            except DocumentEncodingError as error:
                if strict:
                    raise
                self._failed[doc_id] = error
                failed_ids.add(doc_id)
                logger.error("Could not index document %r: %s", doc_id, error)
                continue
            plans.append((doc_id, text, plan))
            changed.append(doc_id)

        if not plans:
            return self._commit([], empty, [doc_id for doc_id in changed if doc_id not in failed_ids])

        try:
            embeddings_by_doc = self._encode_batch(plans)
            records = [
                self._make_record(doc_id, text, plan, embeddings_by_doc[doc_id]) for doc_id, text, plan in plans
            ]
        except DocumentEncodingError:
            if strict:
                # Nothing is committed: the live index is untouched, so a failure cannot
                # leave a mix of old and new entries.
                raise
            # Find out which documents can still be committed, one at a time.
            records = []
            for doc_id, text, plan in plans:
                try:
                    records.append(self._make_record(doc_id, text, plan, self._encode_chunks(doc_id, plan)))
                except DocumentEncodingError as document_error:
                    self._failed[doc_id] = document_error
                    failed_ids.add(doc_id)
                    logger.error("Could not index document %r: %s", doc_id, document_error)

        # A document whose encoding failed is not "changed", even though it was attempted.
        return self._commit(records, empty, [doc_id for doc_id in changed if doc_id not in failed_ids])

    def _commit(
        self,
        records: Sequence[_DocumentRecord],
        empty: Sequence[str],
        changed: Sequence[str],
    ) -> list[str]:
        for record in records:
            self._records[record.plan.doc_id] = record
            self._failed.pop(record.plan.doc_id, None)
            self._skipped.pop(record.plan.doc_id, None)
        for doc_id in empty:
            had_record = self._records.pop(doc_id, None) is not None
            self._failed.pop(doc_id, None)
            self._skipped[doc_id] = "empty"
            logger.warning(
                "Document %r is empty or whitespace-only and was not indexed%s.",
                doc_id,
                "; its previous index entry was removed" if had_record else "",
            )

        if changed:
            logger.info(
                "DocumentRetriever changed %d document(s); %d chunk(s) in the index.",
                len(changed),
                self.num_chunks,
            )
        return list(changed)

    def remove_documents(self, doc_ids: Iterable[str]) -> list[str]:
        """Remove documents from the index.

        Args:
            doc_ids: Identifiers to remove. Unknown identifiers are ignored.

        Returns:
            list[str]: The identifiers that were present and have been removed.
        """
        removed = []
        for doc_id in doc_ids:
            record = self._records.pop(doc_id, None)
            self._skipped.pop(doc_id, None)
            self._failed.pop(doc_id, None)
            if record is not None:
                removed.append(doc_id)
        return removed

    def update_document(self, doc_id: str, text: str, *, strict: bool = True) -> bool:
        """Update one document, re-chunking only it.

        Args:
            doc_id: Identifier of the document.
            text: The new text.
            strict: Raise on encoding failure instead of recording it. Defaults to ``True``.

        Returns:
            bool: Whether the document was re-encoded. ``False`` means the text was
            unchanged or empty, or (with ``strict=False``) encoding failed.
        """
        return doc_id in self.add_documents({doc_id: text}, strict=strict)

    # ------------------------------------------------------------------ search

    def _aggregate(self, scores: np.ndarray) -> float:
        if self.aggregation == "max":
            return float(scores.max())
        if self.aggregation == "mean_top_n":
            top = np.sort(scores)[::-1][: min(self.mean_top_n, len(scores))]
            return float(top.mean())
        hits = int((scores >= self.hit_threshold).sum())
        # Move the best chunk score towards the top of the similarity range in proportion
        # to how much of the document matched. This keeps the best chunk's ordering, raises
        # documents with more hits above equally good ones with fewer hits, stays bounded,
        # and never inverts when the best score is negative (which a multiplier would).
        best = float(scores.max())
        weight = hits / len(scores)
        return float(best + (1.0 - best) * weight)

    def _snippet(self, chunk: TextChunk) -> str:
        text = chunk.text.strip()
        if len(text) <= self.snippet_max_chars:
            return text
        return text[: self.snippet_max_chars].rstrip() + "…"

    def _build_hit(self, doc_id: str, record: _DocumentRecord, chunk_scores: np.ndarray) -> DocumentHit:
        num_chunks = len(record)
        order = np.argsort(-chunk_scores, kind="stable")[: min(self.max_chunks_per_result, num_chunks)]
        matched = tuple(record.plan.chunks[index] for index in order)
        matched_scores = tuple(float(chunk_scores[index]) for index in order)
        hit_flags = chunk_scores >= self.hit_threshold
        # Coverage answers "how much of this document do the chunks returned in this hit
        # account for?", so it is computed over the attached matched chunks, not over every
        # chunk above the threshold. That keeps it consistent with the cited passages and
        # bounded by max_chunks_per_result.
        matched_flags = [chunk_scores[index] >= self.hit_threshold for index in order]
        matched_plan = ChunkPlan(
            doc_id=record.plan.doc_id,
            text_length=record.plan.text_length,
            chunks=matched,
            total_chunks=record.plan.total_chunks,
        )

        return DocumentHit(
            doc_id=doc_id,
            score=self._aggregate(chunk_scores),
            matched_chunks=matched,
            chunk_scores=matched_scores,
            snippets=tuple(self._snippet(chunk) for chunk in matched),
            coverage=_covered_fraction(matched_plan, matched_flags),
            num_chunks=num_chunks,
            hit_count=int(hit_flags.sum()),
            aggregation=self.aggregation,
        )

    def _flatten_chunk_embeddings(self) -> tuple[Tensor, np.ndarray]:
        """Stack all chunk embeddings and remember the document index of each row."""
        rows: list[Tensor] = []
        doc_of_row: list[int] = []
        for doc_index, record in enumerate(self._records.values()):
            if not len(record):
                continue
            rows.append(record.embeddings)
            doc_of_row.extend([doc_index] * len(record))
        if not rows:
            return torch.empty((0, 0), dtype=torch.float32), np.empty((0,), dtype=np.int64)
        return torch.cat(rows, dim=0), np.asarray(doc_of_row, dtype=np.int64)

    def _search_chunked(self, query_embeddings: Tensor, top_k: int) -> list[list[DocumentHit]]:
        chunk_embeddings, doc_of_row = self._flatten_chunk_embeddings()
        if chunk_embeddings.shape[0] == 0:
            return [[] for _ in range(len(query_embeddings))]

        doc_ids = list(self._records)
        records = list(self._records.values())
        # Grouping row indices by document once keeps the per-query work proportional to
        # the number of chunk scores instead of rescanning the whole matrix per document.
        row_indices_by_doc: dict[int, np.ndarray] = {
            doc_index: np.flatnonzero(doc_of_row == doc_index)
            for doc_index in range(len(records))
            if len(records[doc_index])
        }

        # Score with the same cos_sim the rest of the library uses, over the same embeddings
        # that are attached to each hit, so a document's score and its cited chunks always
        # come from one source. Query chunks bound the transient [queries, chunks] matrix.
        all_scores: list[np.ndarray] = []
        for start in range(0, len(query_embeddings), _QUERY_CHUNK_SIZE):
            chunk = query_embeddings[start : start + _QUERY_CHUNK_SIZE]
            all_scores.append(cos_sim(chunk, chunk_embeddings).cpu().numpy())

        results: list[list[DocumentHit]] = []
        for scores in all_scores:
            for row in scores:
                hits = [
                    self._build_hit(doc_ids[doc_index], records[doc_index], row[row_indices])
                    for doc_index, row_indices in row_indices_by_doc.items()
                ]
                hits.sort(key=lambda hit: hit.score, reverse=True)
                results.append(hits[:top_k])
        return results

    def _search_whole_documents(
        self,
        num_queries: int,
        query_embeddings: Tensor,
        top_k: int,
        score_function: Callable[[Tensor, Tensor], Tensor] | None,
    ) -> list[list[DocumentHit]]:
        """The ``enabled=False`` path: exactly one whole-document embedding per document."""
        doc_ids = list(self._records)
        if not doc_ids:
            return [[] for _ in range(num_queries)]
        corpus = torch.cat([self._records[doc_id].embeddings[:1] for doc_id in doc_ids], dim=0)
        search_kwargs: dict[str, Any] = {"top_k": top_k}
        if score_function is not None:
            search_kwargs["score_function"] = score_function
        raw_hits = semantic_search(query_embeddings, corpus, **search_kwargs)

        results: list[list[DocumentHit]] = []
        for raw in raw_hits:
            hits = []
            for hit in raw:
                doc_id = doc_ids[hit["corpus_id"]]
                record = self._records[doc_id]
                hits.append(
                    DocumentHit(
                        doc_id=doc_id,
                        score=float(hit["score"]),
                        matched_chunks=(),
                        chunk_scores=(),
                        snippets=(),
                        coverage=0.0,
                        num_chunks=len(record),
                        hit_count=0,
                        aggregation="disabled",
                    )
                )
            results.append(hits)
        return results

    @torch.inference_mode()
    def search(
        self,
        queries: str | Sequence[str],
        top_k: int = 10,
        *,
        score_function: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> list[list[DocumentHit]]:
        """Search the indexed documents and return document-level results.

        Args:
            queries: One query string or a sequence of query strings.
            top_k: Maximum number of documents returned per query.
            score_function: Optional scoring function used by the ``enabled=False``
                path. Ignored when chunking is enabled, which always scores chunks
                with cosine similarity (the default of ``semantic_search``).

        Returns:
            list[list[DocumentHit]]: One list of hits per query, best first, with at
            most one hit per document.
        """
        query_list = [queries] if isinstance(queries, str) else list(queries)
        if not query_list:
            return []
        query_embeddings = self._encode_texts(query_list)

        if not self.enabled:
            return self._search_whole_documents(len(query_list), query_embeddings, top_k, score_function)

        return self._search_chunked(query_embeddings, top_k)

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Persist chunk plans, chunk texts and chunk embeddings to a directory.

        The whole index is written to one temporary file that is then atomically moved
        into place, so an interrupted save never leaves a half-written or mixed index.

        Args:
            directory: Target directory. Created if missing.

        Returns:
            Path: The written index file.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": INDEX_FORMAT_VERSION,
            "config": {
                "max_chars": self.config.max_chars,
                "overlap_chars": self.config.overlap_chars,
                "max_chunks_per_doc": self.config.max_chunks_per_doc,
                "split_oversized_sentences": self.config.split_oversized_sentences,
                "allow_hard_split": self.config.allow_hard_split,
                "sentence_terminators": self.config.sentence_terminators,
                "clause_terminators": self.config.clause_terminators,
            },
            "documents": [
                {
                    "doc_id": doc_id,
                    "text": record.text,
                    "text_hash": record.text_hash,
                    "plan": record.plan.to_dict(),
                }
                for doc_id, record in self._records.items()
            ],
            "embeddings": {doc_id: record.embeddings for doc_id, record in self._records.items()},
            "skipped": dict(self._skipped),
        }

        final_path = target / "index.pt"
        tmp_path = target / f".index.pt.tmp-{os.getpid()}"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, final_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return final_path

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        model: SentenceTransformer,
        **kwargs: Any,
    ) -> DocumentRetriever:
        """Restore an index written by :meth:`save`.

        Chunk plans and chunk embeddings are restored verbatim, so loading does not
        re-chunk or re-encode anything.

        Args:
            directory: Directory written by :meth:`save`.
            model: Model used to encode future queries and new documents.
            **kwargs: Search/aggregation overrides, e.g. ``aggregation="max"``. Chunking
                parameters are restored from the index unless ``config`` is passed.

        Returns:
            DocumentRetriever: A retriever with the saved chunks and embeddings.
        """
        source = Path(directory)
        payload = torch.load(source / "index.pt", map_location="cpu", weights_only=True)
        if payload.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported index format version {payload.get('format_version')!r}; expected {INDEX_FORMAT_VERSION}"
            )
        config = kwargs.pop("config", None)
        if config is None:
            config = ChunkingConfig(**payload["config"])

        retriever = cls(model, config, **kwargs)
        embeddings = payload["embeddings"]
        for document in payload["documents"]:
            doc_id = document["doc_id"]
            doc_embeddings = embeddings[doc_id]
            plan = _plan_from_dict(document["plan"], document["text"])
            if len(plan.chunks) != len(doc_embeddings):
                raise ValueError(
                    f"Index for document {doc_id!r} is inconsistent: {len(plan.chunks)} chunks but "
                    f"{len(doc_embeddings)} embeddings. Rebuild the index."
                )
            retriever._records[doc_id] = _DocumentRecord(
                text_hash=document["text_hash"],
                text=document["text"],
                plan=plan,
                embeddings=doc_embeddings,
            )
        retriever._skipped.update(payload.get("skipped", {}))
        return retriever


def _plan_from_dict(plan_dict: Mapping[str, Any], text: str) -> ChunkPlan:
    """Rebuild a :class:`ChunkPlan` from a saved manifest, validating its spans."""
    chunks = []
    for raw in plan_dict["chunks"]:
        start, end = raw["start_char"], raw["end_char"]
        if not (0 <= start < end <= len(text)) or text[start:end] != raw.get("text", text[start:end]):
            raise ValueError(
                f"Saved chunk {raw['chunk_index']} of document {plan_dict['doc_id']!r} does not match its text; "
                "the index is corrupted."
            )
        chunks.append(
            TextChunk(
                doc_id=raw["doc_id"],
                chunk_index=raw["chunk_index"],
                start_char=start,
                end_char=end,
                text=text[start:end],
                end_boundary=raw["end_boundary"],
                forced=raw.get("forced", False),
            )
        )
    truncation = plan_dict.get("truncation")
    if truncation is not None:
        from .chunking import TruncationInfo

        truncation = TruncationInfo(
            total_chunks=truncation["total_chunks"],
            kept_chunks=truncation["kept_chunks"],
            dropped_chunks=truncation["dropped_chunks"],
            dropped_start_char=truncation["dropped_start_char"],
            dropped_end_char=truncation["dropped_end_char"],
        )
    return ChunkPlan(
        doc_id=plan_dict["doc_id"],
        text_length=plan_dict["text_length"],
        chunks=tuple(chunks),
        total_chunks=plan_dict["total_chunks"],
        truncation=truncation,
        forced_chunks=plan_dict.get("forced_chunks", 0),
    )


class CachedChunkEncoder:
    """Reusable chunk plan + chunk embeddings for a corpus of documents.

    This is a thin wrapper around :class:`DocumentRetriever` that makes the
    "chunk once, query many times" workflow explicit. It delegates all caching,
    aggregation and persistence to the retriever.

    Args:
        retriever: The retriever that owns the index.
    """

    def __init__(self, retriever: DocumentRetriever) -> None:
        self.retriever = retriever

    def build(self, documents: Mapping[str, str], *, strict: bool = True) -> list[str]:
        """Index (or update) the documents and return the identifiers that were encoded."""
        return self.retriever.add_documents(documents, strict=strict)

    def search(self, queries: str | Sequence[str], top_k: int = 10) -> list[list[DocumentHit]]:
        """Search without re-chunking or re-encoding the corpus."""
        return self.retriever.search(queries, top_k=top_k)
