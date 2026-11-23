"""Semantic-boundary text chunking for long-document retrieval.

The chunker splits a document into contiguous character spans whose boundaries
always fall between sentences (and between clauses/words only when a single
sentence cannot fit into one chunk). Every chunk keeps the absolute character
range it came from so retrieval results can be traced back to the source text.

The implementation is intentionally deterministic: the same text and
:class:`ChunkingConfig` always produce the same chunks, character for
character. Documents that fit into a single chunk are returned as one chunk
that is identical to the original document, so enabling chunking does not
change how short documents are treated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

Boundary = Literal["sentence", "clause", "word", "hard", "end"]


class ChunkingError(ValueError):
    """Raised when a document cannot be chunked under the configured policy."""

    def __init__(self, doc_id: str, position: int, message: str) -> None:
        self.doc_id = doc_id
        self.position = position
        super().__init__(f"Could not chunk document {doc_id!r} at character {position}: {message}")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Parameters controlling how documents are split into chunks.

    Args:
        max_chars: Maximum number of characters in a chunk. A chunk never ends in the
            middle of a sentence unless the sentence is longer than this value and no
            clause or word boundary is available (see ``allow_hard_split``).
        overlap_chars: Approximate number of characters shared by two consecutive
            chunks. The overlap is rounded up to the nearest whole sentence (or
            clause/word) boundary, so the actual overlap can be slightly larger.
        max_chunks_per_doc: Maximum number of chunks kept for one document. Documents
            that need more chunks are truncated and reported through
            :class:`ChunkPlan`.
        split_oversized_sentences: When a single sentence is longer than ``max_chars``,
            split it at clause punctuation first and then at whitespace. When
            ``False``, such a document raises :class:`ChunkingError` instead.
        allow_hard_split: Last-resort fallback for text without any punctuation or
            whitespace. When ``True``, an oversized run is split every ``max_chars``
            characters and the resulting chunks are marked with boundary ``"hard"``.
            When ``False``, :class:`ChunkingError` is raised instead.
        sentence_terminators: Characters that end a sentence. The default covers
            common CJK and ASCII sentence punctuation.
        clause_terminators: Characters that end a clause, used only for sentences
            longer than ``max_chars``.
    """

    max_chars: int = 1000
    overlap_chars: int = 100
    max_chunks_per_doc: int = 512
    split_oversized_sentences: bool = True
    allow_hard_split: bool = True
    sentence_terminators: str = "。！？!?；;…"
    clause_terminators: str = "，,、：:"

    def __post_init__(self) -> None:
        if not isinstance(self.max_chars, int) or self.max_chars < 1:
            raise ValueError(f"max_chars must be a positive integer, got {self.max_chars!r}")
        if not isinstance(self.overlap_chars, int) or self.overlap_chars < 0:
            raise ValueError(f"overlap_chars must be a non-negative integer, got {self.overlap_chars!r}")
        if self.overlap_chars >= self.max_chars:
            raise ValueError(f"overlap_chars ({self.overlap_chars}) must be smaller than max_chars ({self.max_chars})")
        if not isinstance(self.max_chunks_per_doc, int) or self.max_chunks_per_doc < 1:
            raise ValueError(f"max_chunks_per_doc must be a positive integer, got {self.max_chunks_per_doc!r}")
        if not self.sentence_terminators:
            raise ValueError("sentence_terminators must not be empty")
        _patterns(self)


@dataclass(frozen=True, slots=True)
class TruncationInfo:
    """Describes how a too-long document was truncated.

    Args:
        total_chunks: Number of chunks the document would have had without the cap.
        kept_chunks: Number of chunks that were kept.
        dropped_chunks: Number of chunks that were dropped.
        dropped_start_char: First character of the dropped tail.
        dropped_end_char: End of the document (exclusive), i.e. the end of the dropped tail.
    """

    total_chunks: int
    kept_chunks: int
    dropped_chunks: int
    dropped_start_char: int
    dropped_end_char: int

    @property
    def dropped_chars(self) -> int:
        """Number of characters in the dropped tail."""
        return self.dropped_end_char - self.dropped_start_char

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable representation."""
        return {
            "total_chunks": self.total_chunks,
            "kept_chunks": self.kept_chunks,
            "dropped_chunks": self.dropped_chunks,
            "dropped_start_char": self.dropped_start_char,
            "dropped_end_char": self.dropped_end_char,
            "dropped_chars": self.dropped_chars,
        }


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One contiguous span of a document.

    Args:
        doc_id: Identifier of the source document.
        chunk_index: Zero-based position of the chunk inside its document.
        start_char: Inclusive start offset in the original document.
        end_char: Exclusive end offset in the original document.
        text: The chunk text, exactly ``document[start_char:end_char]``.
        end_boundary: What kind of boundary the chunk ends on: a sentence, a clause,
            a word, a forced hard cut, or the end of the document.
        forced: Whether the chunk boundary is a last-resort hard cut.
    """

    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    end_boundary: Boundary
    forced: bool = False

    def to_dict(self) -> dict[str, str | int | bool]:
        """Return a JSON-serialisable representation without the chunk text."""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "end_boundary": self.end_boundary,
            "forced": self.forced,
        }


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """The result of planning the chunks of one document.

    Args:
        doc_id: Identifier of the source document.
        text_length: Length of the original document in characters.
        chunks: The chunks that will be indexed, in order.
        total_chunks: Number of chunks before ``max_chunks_per_doc`` truncation.
        truncation: Details about the dropped tail, or ``None``.
        forced_chunks: Number of chunks whose end boundary is a forced hard cut.
    """

    doc_id: str
    text_length: int
    chunks: tuple[TextChunk, ...]
    total_chunks: int
    truncation: TruncationInfo | None = None
    forced_chunks: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether the document has no chunk (e.g. it is empty or whitespace only)."""
        return not self.chunks

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "doc_id": self.doc_id,
            "text_length": self.text_length,
            "total_chunks": self.total_chunks,
            "truncation": self.truncation.to_dict() if self.truncation else None,
            "forced_chunks": self.forced_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    boundary: Boundary


@lru_cache(maxsize=32)
def _patterns(config: ChunkingConfig) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    closers = r"[\"'”’」』】）)\]]*"
    sentence_terminators = re.escape(config.sentence_terminators)
    clause_terminators = re.escape(config.clause_terminators)
    sentence = re.compile(
        rf"(?:[{sentence_terminators}]+{closers}|\n{{2,}}|\n(?=[#*\-•]|\d+[.)])|\.(?=\s|$))", re.UNICODE
    )
    clause = re.compile(rf"(?:[{clause_terminators}]+{closers}|\n+|[—–]+|--+)", re.UNICODE)
    whitespace = re.compile(r"\s+", re.UNICODE)
    return sentence, clause, whitespace


def _split_spans(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` after every pattern match, returning absolute spans."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(text, start, end):
        match_end = min(match.end(), end)
        if match_end <= cursor:
            continue
        spans.append((cursor, match_end))
        cursor = match_end
    if cursor < end:
        spans.append((cursor, end))
    if not spans:
        return [(start, end)]

    merged: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        # Whitespace between two sentences belongs to the preceding unit so that a chunk
        # never starts with a blank line or an indentation run. Leading whitespace of the
        # document is kept in the first unit so the units still cover the whole text.
        if merged:
            while span_start < span_end and text[span_start].isspace():
                span_start += 1
        if span_start >= span_end:
            if merged:
                merged[-1] = (merged[-1][0], span_end)
            continue
        if merged:
            merged[-1] = (merged[-1][0], span_start)
        merged.append((span_start, span_end))
    return merged or [(start, end)]


def _hard_spans(start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    return [(offset, min(offset + max_chars, end)) for offset in range(start, end, max_chars)]


def _atomic_units(doc_id: str, text: str, config: ChunkingConfig) -> list[_Unit]:
    """Return units no longer than ``max_chars`` that partition the document."""
    sentence_pattern, clause_pattern, whitespace_pattern = _patterns(config)
    units: list[_Unit] = []

    def add_oversized(start: int, end: int) -> None:
        if not config.split_oversized_sentences:
            raise ChunkingError(
                doc_id, start, f"a {end - start}-character sentence exceeds max_chars={config.max_chars}"
            )
        for clause_start, clause_end in _split_spans(text, start, end, clause_pattern):
            if clause_end - clause_start <= config.max_chars:
                units.append(_Unit(clause_start, clause_end, "clause"))
                continue
            for word_start, word_end in _split_spans(text, clause_start, clause_end, whitespace_pattern):
                if word_end - word_start <= config.max_chars:
                    units.append(_Unit(word_start, word_end, "word"))
                    continue
                if not config.allow_hard_split:
                    raise ChunkingError(
                        doc_id,
                        word_start,
                        "text without a sentence, clause or word boundary exceeds max_chars and hard splitting "
                        "is disabled",
                    )
                for hard_start, hard_end in _hard_spans(word_start, word_end, config.max_chars):
                    units.append(_Unit(hard_start, hard_end, "hard"))

    for sentence_start, sentence_end in _split_spans(text, 0, len(text), sentence_pattern):
        if sentence_end - sentence_start <= config.max_chars:
            units.append(_Unit(sentence_start, sentence_end, "sentence"))
        else:
            add_oversized(sentence_start, sentence_end)
    return units


def _build_chunks(doc_id: str, text: str, config: ChunkingConfig, units: list[_Unit]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    start_index = 0
    chunk_index = 0
    while start_index < len(units):
        end_index = start_index + 1
        while end_index < len(units) and units[end_index].end - units[start_index].start <= config.max_chars:
            end_index += 1

        chunk_start = units[start_index].start
        chunk_end = units[end_index - 1].end
        boundary: Boundary = "end" if chunk_end == len(text) else units[end_index - 1].boundary
        chunks.append(
            TextChunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                start_char=chunk_start,
                end_char=chunk_end,
                text=text[chunk_start:chunk_end],
                end_boundary=boundary,
                forced=units[end_index - 1].boundary == "hard" and chunk_end != len(text),
            )
        )
        chunk_index += 1
        if end_index >= len(units):
            break

        # Start the next chunk inside the trailing units of this one, re-using whole units
        # until the overlap reaches ``overlap_chars``. Rounding to a unit boundary keeps
        # the next chunk's start out of the middle of a sentence. Always advance by at
        # least one unit so a single oversized unit cannot make the loop repeat itself.
        next_index = end_index
        if config.overlap_chars > 0 and end_index - start_index > 1:
            # Always start the next chunk at least one unit back, then extend the overlap
            # backwards while it is still shorter than requested.
            next_index = end_index - 1
            while (
                next_index - 1 > start_index
                and units[end_index - 1].end - units[next_index - 1].start < config.overlap_chars
            ):
                next_index -= 1
        start_index = max(next_index, start_index + 1)

    return chunks


def chunk_document(doc_id: str, text: str, config: ChunkingConfig | None = None) -> ChunkPlan:
    """Split one document into chunks aligned with sentence boundaries.

    An empty or whitespace-only document yields an empty plan (no chunk) so callers
    can report it explicitly instead of indexing a meaningless vector. A document
    that fits in one chunk is returned as a single chunk whose text is identical to
    the input, which keeps the behaviour of short documents unchanged.

    Args:
        doc_id: Identifier of the document, copied into every chunk.
        text: The document text.
        config: Chunking parameters. Defaults to ``ChunkingConfig()``.

    Returns:
        ChunkPlan: The chunks, plus truncation and forced-cut information.

    Raises:
        TypeError: If ``text`` is not a string.
        ChunkingError: If the document needs a split that the configuration forbids.
    """
    config = config or ChunkingConfig()
    if not isinstance(doc_id, str):
        raise TypeError(f"doc_id must be a string, got {type(doc_id).__name__}")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    if not text.strip():
        return ChunkPlan(doc_id=doc_id, text_length=len(text), chunks=(), total_chunks=0)

    if len(text) <= config.max_chars:
        chunk = TextChunk(
            doc_id=doc_id,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            text=text,
            end_boundary="end",
        )
        return ChunkPlan(doc_id=doc_id, text_length=len(text), chunks=(chunk,), total_chunks=1)

    units = _atomic_units(doc_id, text, config)
    chunks = _build_chunks(doc_id, text, config, units)

    total_chunks = len(chunks)
    truncation = None
    if total_chunks > config.max_chunks_per_doc:
        kept = chunks[: config.max_chunks_per_doc]
        dropped = chunks[config.max_chunks_per_doc :]
        # The dropped chunk may overlap the last kept one; only the part beyond the last
        # kept character is truly unreachable, so report the range that is really lost.
        lost_start = max(kept[-1].end_char, dropped[0].start_char)
        truncation = TruncationInfo(
            total_chunks=total_chunks,
            kept_chunks=len(kept),
            dropped_chunks=len(dropped),
            dropped_start_char=lost_start,
            dropped_end_char=len(text),
        )
        logger.warning(
            "Document %r produced %d chunks, which exceeds max_chunks_per_doc=%d; keeping chunks 0-%d and "
            "dropping %d chunk(s) covering characters [%d, %d) (%d characters). Content after that range "
            "cannot be retrieved.",
            doc_id,
            total_chunks,
            config.max_chunks_per_doc,
            config.max_chunks_per_doc - 1,
            truncation.dropped_chunks,
            truncation.dropped_start_char,
            truncation.dropped_end_char,
            truncation.dropped_chars,
        )
        chunks = kept

    forced_chunks = sum(1 for chunk in chunks if chunk.forced)
    if forced_chunks:
        logger.warning(
            "Document %r has %d chunk(s) that had to be split inside a sentence because no sentence, clause or "
            "word boundary was available within max_chars=%d. Retrieval quality for those spans may be lower.",
            doc_id,
            forced_chunks,
            config.max_chars,
        )

    return ChunkPlan(
        doc_id=doc_id,
        text_length=len(text),
        chunks=tuple(chunks),
        total_chunks=total_chunks,
        truncation=truncation,
        forced_chunks=forced_chunks,
    )


def chunk_text(text: str, config: ChunkingConfig | None = None, doc_id: str = "") -> list[TextChunk]:
    """Convenience wrapper around :func:`chunk_document` that returns only the chunks.

    Args:
        text: The document text.
        config: Chunking parameters. Defaults to ``ChunkingConfig()``.
        doc_id: Identifier copied into every chunk. Defaults to an empty string.

    Returns:
        list[TextChunk]: The chunks in order. Empty for an empty document.
    """
    return list(chunk_document(doc_id, text, config).chunks)
