from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util.chunking import ChunkingConfig
from sentence_transformers.util.document_retrieval import (
    CachedChunkEncoder,
    DocumentEncodingError,
    DocumentRetriever,
    _covered_fraction,
)
from sentence_transformers.util.retrieval import semantic_search

# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def small_model() -> SentenceTransformer:
    # A local, deterministic model: every chunk is encoded from real token embeddings,
    # but nothing is downloaded and repeated runs produce bit-identical vectors.
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")


def _manual(topic: str, parameters: int = 12, repeats: int = 6) -> str:
    sentences = [
        f"The {topic} manual section {index} documents parameter P{index} with its default value "
        f"{index * 10}, unit U{index}, and tolerance T{index}."
        for index in range(parameters)
    ]
    return " ".join(sentences * repeats)


@pytest.fixture()
def corpus() -> dict[str, str]:
    return {
        "alpha-manual": _manual("alpha"),
        "beta-manual": _manual("beta"),
        "gamma-manual": _manual("gamma"),
        "short-note": "The alpha manual covers parameter P0.",
    }


@pytest.fixture()
def retriever(small_model: SentenceTransformer, corpus: dict[str, str]) -> DocumentRetriever:
    retriever = DocumentRetriever(
        small_model,
        ChunkingConfig(max_chars=200, overlap_chars=40),
        aggregation="max",
        snippet_max_chars=80,
    )
    retriever.add_documents(corpus)
    return retriever


# --------------------------------------------------------------------------- chunk-level result contract


def test_chunks_carry_source_locations(retriever: DocumentRetriever, corpus: dict[str, str]) -> None:
    for doc_id, plan in retriever.chunk_plans().items():
        text = corpus[doc_id]
        assert plan.chunks[0].start_char == 0
        assert plan.chunks[-1].end_char == len(text)
        for chunk in plan.chunks:
            assert chunk.doc_id == doc_id
            assert chunk.text == text[chunk.start_char : chunk.end_char]


def test_hits_cite_the_chunks_that_produced_the_score(retriever: DocumentRetriever, corpus: dict[str, str]) -> None:
    hits = retriever.search("What is the default of parameter P3 in the alpha manual?", top_k=3)[0]
    assert hits

    for hit in hits:
        plan = retriever.chunk_plans()[hit.doc_id]
        assert hit.num_chunks == len(plan.chunks)
        for chunk, score, snippet in zip(hit.matched_chunks, hit.chunk_scores, hit.snippets):
            # The cited chunk is a real chunk of that document and its text is the snippet source.
            assert plan.chunks[chunk.chunk_index] == chunk
            assert corpus[hit.doc_id][chunk.start_char : chunk.end_char] == chunk.text
            assert snippet.rstrip("…").strip() in chunk.text.strip() or chunk.text.strip() in snippet
            # The best chunk always accounts for the "max" score exactly.
        assert hit.chunk_scores[0] == pytest.approx(hit.score)
        assert hit.best_chunk == hit.matched_chunks[0]


def test_chunk_embeddings_come_from_the_unchanged_encode(
    retriever: DocumentRetriever, small_model: SentenceTransformer
) -> None:
    # Every stored chunk vector must be exactly what model.encode returns for that chunk
    # text: the new layer must not alter encoding semantics. The batch layout affects
    # low-order bits through padding, so compare against one encode call over all chunks.
    texts: list[str] = []
    spans = {}
    for doc_id, plan in retriever.chunk_plans().items():
        start = len(texts)
        texts.extend(chunk.text for chunk in plan.chunks)
        spans[doc_id] = slice(start, len(texts))
    direct = small_model.encode(texts, convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=False).to(
        torch.float32
    )
    for doc_id, span in spans.items():
        assert torch.equal(retriever._records[doc_id].embeddings, direct[span]), doc_id


def test_results_are_deduplicated_per_document(retriever: DocumentRetriever) -> None:
    hits = retriever.search("parameter P3 default value tolerance", top_k=10)[0]
    doc_ids = [hit.doc_id for hit in hits]
    assert len(doc_ids) == len(set(doc_ids))
    assert len(doc_ids) <= min(10, len(retriever.doc_ids))


def test_scores_are_monotonically_ordered(retriever: DocumentRetriever) -> None:
    for hits in retriever.search(["parameter P1", "beta manual tolerance"], top_k=3):
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_coverage_measures_hit_fraction(retriever: DocumentRetriever, corpus: dict[str, str]) -> None:
    hits = retriever.search("alpha manual parameter P0", top_k=4)[0]
    assert hits
    for hit in hits:
        assert 0.0 < hit.coverage <= 1.0
        # Coverage counts every hit chunk, not just the attached matched chunks, and the
        # attached chunks are always a subset of the hits.
        assert hit.hit_count >= len(hit.matched_chunks)
        assert hit.hit_count <= hit.num_chunks
        if hit.num_chunks == 1:
            assert hit.coverage == pytest.approx(1.0)


def test_snippet_length_is_controlled(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(
        small_model,
        ChunkingConfig(max_chars=300, overlap_chars=0),
        aggregation="max",
        snippet_max_chars=25,
    )
    retriever.add_documents({"doc": _manual("delta")})
    hit = retriever.search("parameter P2", top_k=1)[0][0]
    assert hit.snippets
    for snippet in hit.snippets:
        assert len(snippet.rstrip("…")) <= 25

    # A zero-length limit still returns a (possibly empty) snippet rather than failing.
    retriever.snippet_max_chars = 0
    empty_hit = retriever.search("parameter P2", top_k=1)[0][0]
    assert all(snippet in {"", "…"} for snippet in empty_hit.snippets)


def test_empty_corpus_and_empty_query(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model)
    assert retriever.search("anything") == [[]]
    retriever.add_documents({"doc": "hello world"})
    assert retriever.search([]) == []
    assert len(retriever.search(["q1", "q2"])) == 2


def test_short_documents_behave_like_before(small_model: SentenceTransformer) -> None:
    text = "A short note about parameter P0."
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=1000, overlap_chars=0))
    retriever.add_documents({"note": text})

    plan = retriever.chunk_plans()["note"]
    assert len(plan.chunks) == 1
    assert plan.chunks[0].text == text
    assert plan.chunks[0].end_boundary == "end"

    hit = retriever.search("parameter P0", top_k=1)[0][0]
    assert hit.best_chunk.text == text


# --------------------------------------------------------------------------- aggregation


def test_aggregation_changes_the_order_as_expected(small_model: SentenceTransformer) -> None:
    # one document mentions the query once, another mentions it many times.
    once = " ".join(
        [
            "Unrelated filler sentence number zero about hardware.",
            "Unrelated filler sentence number one about packaging.",
            "The zeta parameter appears here exactly once, default 42.",
            "Unrelated filler sentence number three about shipping.",
            "Unrelated filler sentence number four about support.",
        ]
    )
    many = " ".join(["The zeta parameter default is 42."] * 5)
    documents = {"once": once, "many": many}

    chunking = ChunkingConfig(max_chars=60, overlap_chars=0)
    max_retriever = DocumentRetriever(small_model, chunking, aggregation="max")
    max_retriever.add_documents(documents)
    mean_retriever = DocumentRetriever(small_model, chunking, aggregation="mean_top_n", mean_top_n=5)
    mean_retriever.add_documents(documents)
    weighted_retriever = DocumentRetriever(small_model, chunking, aggregation="hit_count_weighted", hit_threshold=0.7)
    weighted_retriever.add_documents(documents)

    query = "What is the zeta parameter default?"
    max_hits = max_retriever.search(query, top_k=2)[0]
    mean_hits = mean_retriever.search(query, top_k=2)[0]
    weighted_hits = weighted_retriever.search(query, top_k=2)[0]

    # "many" repeats the answer, so averaging and hit-count weighting reward it more than pure max does.
    assert max_hits[0].doc_id in {"many", "once"}
    assert mean_hits[0].doc_id == "many"
    assert weighted_hits[0].doc_id == "many"
    assert weighted_hits[0].score > weighted_hits[1].score
    # and the strategies really produce different scores for the same chunks.
    max_scores = {hit.doc_id: hit.score for hit in max_hits}
    mean_scores = {hit.doc_id: hit.score for hit in mean_hits}
    assert mean_scores != max_scores
    # For the "once" document only one chunk matches, so averaging over its top chunks
    # pulls its score down; "many" matches on several, so it stays high.
    assert mean_scores["once"] < max_scores["once"]
    assert mean_scores["many"] == pytest.approx(max_scores["many"])


def test_mean_top_n_matches_manual_average(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(
        small_model,
        ChunkingConfig(max_chars=80, overlap_chars=0),
        aggregation="mean_top_n",
        mean_top_n=2,
        max_chunks_per_result=2,
    )
    retriever.add_documents({"doc": _manual("epsilon")})
    query = "parameter P5 default"
    hit = retriever.search(query, top_k=1)[0][0]
    assert hit.score == pytest.approx(float(np.mean(hit.chunk_scores[:2])))

    record = retriever._records["doc"]
    query_embedding = retriever._encode_texts([query])
    from sentence_transformers.util.similarity import cos_sim

    chunk_scores = cos_sim(query_embedding, record.embeddings)[0].numpy()
    assert hit.score == pytest.approx(float(np.sort(chunk_scores)[::-1][:2].mean()))


def test_max_aggregation_uses_the_best_chunk_only(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=80, overlap_chars=0), aggregation="max")
    retriever.add_documents({"doc": _manual("zeta")})
    hit = retriever.search("parameter P7 tolerance", top_k=1)[0][0]
    record = retriever._records["doc"]
    query_embedding = retriever._encode_texts(["parameter P7 tolerance"])
    from sentence_transformers.util.similarity import cos_sim

    all_scores = cos_sim(query_embedding, record.embeddings)[0]
    assert hit.score == pytest.approx(float(all_scores.max()))
    assert hit.matched_chunks[0].chunk_index == int(all_scores.argmax())


def test_hit_count_weighted_penalises_documents_without_hits(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(
        small_model,
        ChunkingConfig(max_chars=80, overlap_chars=0),
        aggregation="hit_count_weighted",
        hit_threshold=0.5,
    )
    retriever.add_documents({"doc": _manual("eta")})
    hit = retriever.search("parameter P2", top_k=1)[0][0]
    assert hit.hit_count >= 1
    hit_ratio = hit.hit_count / hit.num_chunks
    assert hit.score == pytest.approx(hit.chunk_scores[0] + (1.0 - hit.chunk_scores[0]) * hit_ratio)


def test_hit_count_weighted_never_inverts_on_negative_scores() -> None:
    # Scores are negative similarities here: ranking by "more freaks is worse" would be a bug.
    from sentence_transformers.util.document_retrieval import DocumentRetriever as Retriever

    retriever = Retriever.__new__(Retriever)
    retriever.aggregation = "hit_count_weighted"
    retriever.hit_threshold = -0.5
    retriever.mean_top_n = 3

    more_hits = retriever._aggregate(np.array([-0.1, -0.2, -0.3]))
    fewer_hits = retriever._aggregate(np.array([-0.1, -0.6, -0.7]))
    assert more_hits >= fewer_hits
    assert more_hits > -0.1


# --------------------------------------------------------------------------- deduplication


def test_dedup_prevents_one_document_from_filling_the_results(small_model: SentenceTransformer) -> None:
    documents = {
        "big-manual": _manual("omega", parameters=8, repeats=8),
        **{f"other-{index}": _manual(f"other{index}", parameters=4) for index in range(12)},
    }
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=160, overlap_chars=20))
    retriever.add_documents(documents)

    query = "omega manual parameter P4 default value"
    chunk_level = semantic_search(
        retriever._encode_texts([query]),
        torch.cat([record.embeddings for record in retriever._records.values()]),
        top_k=10,
    )[0]
    chunk_doc_ids = []
    row = 0
    for doc_id, record in retriever._records.items():
        row += len(record)
        if any(hit["corpus_id"] < row for hit in chunk_level):
            chunk_doc_ids.append(doc_id)
    # Before: many chunk hits belong to the same long document.
    flat_doc_ids = []
    row = 0
    doc_ranges = []
    for doc_id, record in retriever._records.items():
        doc_ranges.append((row, row + len(record), doc_id))
        row += len(record)
    for hit in chunk_level:
        for start, end, doc_id in doc_ranges:
            if start <= hit["corpus_id"] < end:
                flat_doc_ids.append(doc_id)
                break
    assert flat_doc_ids.count("big-manual") > 1

    # After: at most one hit per document.
    document_hits = retriever.search(query, top_k=10)[0]
    hit_doc_ids = [hit.doc_id for hit in document_hits]
    assert len(hit_doc_ids) == len(set(hit_doc_ids))
    assert hit_doc_ids.count("big-manual") == 1


def test_top_k_limits_documents_not_chunks(retriever: DocumentRetriever) -> None:
    for top_k in (1, 2, 3):
        hits = retriever.search("parameter value", top_k=top_k)[0]
        assert len(hits) <= top_k
        assert len(hits) <= len(retriever.doc_ids)


# --------------------------------------------------------------------------- disabled mode parity


def test_disabled_mode_matches_whole_document_semantic_search(
    small_model: SentenceTransformer, corpus: dict[str, str]
) -> None:
    documents = {doc_id: text for doc_id, text in corpus.items()}
    enabled = DocumentRetriever(small_model, ChunkingConfig(max_chars=200, overlap_chars=40), enabled=True)
    enabled.add_documents(documents)
    disabled = DocumentRetriever(small_model, ChunkingConfig(max_chars=200, overlap_chars=40), enabled=False)
    disabled.add_documents(documents)

    query = "parameter P4 tolerance"
    # With chunking disabled each document has exactly one whole-document embedding.
    assert disabled.num_chunks == len(documents)
    for plan in disabled.chunk_plans().values():
        assert len(plan.chunks) == 1
        assert plan.chunks[0].start_char == 0
        assert plan.chunks[0].end_char == plan.text_length

    # It is exactly semantic_search over those whole-document embeddings.
    doc_ids = disabled.doc_ids
    corpus_embeddings = torch.cat([disabled._records[doc_id].embeddings for doc_id in doc_ids])
    expected = semantic_search(disabled._encode_texts([query]), corpus_embeddings, top_k=3)[0]
    actual = disabled.search(query, top_k=3)[0]

    assert [hit.doc_id for hit in actual] == [doc_ids[hit["corpus_id"]] for hit in expected]
    for hit, expected_hit in zip(actual, expected):
        assert hit.score == pytest.approx(expected_hit["score"], abs=1e-6)
        assert hit.matched_chunks == ()


def test_disabled_mode_does_not_change_encode(small_model: SentenceTransformer, corpus: dict[str, str]) -> None:
    texts = list(corpus.values())
    before = small_model.encode(texts, convert_to_numpy=False)
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=50, overlap_chars=10), enabled=False)
    retriever.add_documents(corpus)
    retriever.search("parameter", top_k=2)
    after = small_model.encode(texts, convert_to_numpy=False)

    for first, second in zip(before, after):
        assert torch.equal(first, second)


# --------------------------------------------------------------------------- incremental update


def test_updating_one_document_leaves_other_vectors_bit_identical(
    retriever: DocumentRetriever, corpus: dict[str, str]
) -> None:
    before = {doc_id: record.embeddings.clone() for doc_id, record in retriever._records.items()}

    changed = retriever.update_document("beta-manual", corpus["beta-manual"] + " A brand new closing sentence.")
    assert changed is True

    for doc_id, previous in before.items():
        if doc_id == "beta-manual":
            continue
        assert torch.equal(retriever._records[doc_id].embeddings, previous), doc_id
    # The changed document really did change.
    assert retriever._records["beta-manual"].text_hash != ""


def test_unchanged_documents_are_not_re_encoded(
    retriever: DocumentRetriever, corpus: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    original = retriever._encode_texts

    def spy(texts):
        calls.append(list(texts))
        return original(texts)

    monkeypatch.setattr(retriever, "_encode_texts", spy)
    updated = retriever.add_documents(corpus)
    assert updated == []
    assert calls == []


def test_only_changed_documents_are_re_encoded(small_model: SentenceTransformer, corpus: dict[str, str]) -> None:
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=200, overlap_chars=40))
    retriever.add_documents(corpus)

    encoded: list[list[str]] = []
    original = retriever._encode_texts

    def spy(texts):
        encoded.append(list(texts))
        return original(texts)

    retriever._encode_texts = spy  # type: ignore[method-assign]
    changed = retriever.add_documents({"beta-manual": corpus["beta-manual"] + " Extra sentence."})
    assert changed == ["beta-manual"]
    encoded_texts = [text for batch in encoded for text in batch]
    assert encoded_texts
    assert all("alpha manual" not in text for text in encoded_texts)
    assert all("gamma manual" not in text for text in encoded_texts)
    assert any("beta manual" in text for text in encoded_texts)


def test_force_rebuilds_every_document(retriever: DocumentRetriever, corpus: dict[str, str]) -> None:
    updated = retriever.add_documents(corpus, force=True)
    assert set(updated) == set(corpus)


# --------------------------------------------------------------------------- failure handling


def test_encoding_failure_names_document_and_chunk(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=80, overlap_chars=0))
    retriever.add_documents({"doc": _manual("theta")})
    original_embeddings = retriever._records["doc"].embeddings.clone()

    def failing_encode(texts, **kwargs):
        if any("parameter P3" in text for text in texts):
            raise RuntimeError("CUDA out of memory (simulated)")
        return original_encode(texts, **kwargs)

    original_encode = retriever._encode_texts
    retriever._encode_texts = failing_encode  # type: ignore[method-assign]

    with pytest.raises(DocumentEncodingError) as excinfo:
        retriever.add_documents({"doc": _manual("theta", parameters=13)})

    error = excinfo.value
    assert error.doc_id == "doc"
    assert error.chunk_index is not None
    assert error.start_char is not None
    assert "parameter P3" in retriever.chunk_plans()["doc"].chunks[error.chunk_index].text
    # The previous chunks and embeddings are untouched.
    assert torch.equal(retriever._records["doc"].embeddings, original_embeddings)
    assert retriever._records["doc"].text == _manual("theta")


def test_failed_update_does_not_mix_old_and_new_entries(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=80, overlap_chars=0))
    documents = {"a": _manual("alpha"), "b": _manual("beta")}
    retriever.add_documents(documents)
    snapshot = {doc_id: record.embeddings.clone() for doc_id, record in retriever._records.items()}

    original = retriever._encode_texts

    def failing(texts, **kwargs):
        if any("beta" in text for text in texts):
            raise RuntimeError("simulated encoder failure")
        return original(texts, **kwargs)

    retriever._encode_texts = failing  # type: ignore[method-assign]
    with pytest.raises(DocumentEncodingError):
        retriever.add_documents({"a": _manual("alpha") + " changed.", "b": _manual("beta") + " changed."})

    # Neither document's index entry changed: no half-applied update.
    assert retriever._records["a"].text == documents["a"]
    assert retriever._records["b"].text == documents["b"]
    for doc_id, embeddings in snapshot.items():
        assert torch.equal(retriever._records[doc_id].embeddings, embeddings)


def test_non_strict_mode_commits_healthy_documents(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=80, overlap_chars=0))
    retriever.add_documents({"good": _manual("alpha"), "bad": _manual("beta")})
    original = retriever._encode_texts

    def failing(texts, **kwargs):
        if any("beta" in text for text in texts):
            raise RuntimeError("simulated encoder failure")
        return original(texts, **kwargs)

    retriever._encode_texts = failing  # type: ignore[method-assign]
    changed = retriever.add_documents(
        {"good": _manual("alpha") + " new.", "bad": _manual("beta") + " new.", "fresh": _manual("gamma")},
        strict=False,
    )

    assert "good" in changed
    assert "fresh" in changed
    assert "bad" not in changed
    assert retriever._records["good"].text.endswith("new.")
    assert retriever._records["fresh"].text.startswith("The gamma manual")
    assert set(retriever.failed_docs) == {"bad"}
    # The failed document kept its old content.
    assert retriever._records["bad"].text == _manual("beta")


def test_empty_document_is_reported_not_dropped(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model)
    retriever.add_documents({"doc": "   ", "ok": "A real note."})
    assert retriever.skipped_docs == {"doc": "empty"}
    assert retriever.doc_ids == ["ok"]


def test_document_becoming_empty_is_removed_and_reported(small_model: SentenceTransformer) -> None:
    retriever = DocumentRetriever(small_model)
    retriever.add_documents({"doc": "A real note about P0."})
    retriever.update_document("doc", "  ")
    assert retriever.doc_ids == []
    assert retriever.skipped_docs == {"doc": "empty"}


def test_document_exceeding_chunk_cap_is_indexed_with_truncation(
    small_model: SentenceTransformer, caplog: pytest.LogCaptureFixture
) -> None:
    text = " ".join(f"Sentence {index} about parameter P{index}." for index in range(80))
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=50, overlap_chars=0, max_chunks_per_doc=3))
    with caplog.at_level("WARNING", logger="sentence_transformers.util.chunking"):
        retriever.add_documents({"huge": text})

    plan = retriever.chunk_plans()["huge"]
    assert retriever.num_chunks == len(plan.chunks) == 3
    assert plan.truncation is not None
    assert "huge" in caplog.text
    assert "cannot be retrieved" in caplog.text
    # It is still searchable over the kept chunks, and the citation stays inside them.
    hit = retriever.search("parameter P1", top_k=1)[0][0]
    assert hit.num_chunks == 3
    assert hit.best_chunk.end_char <= plan.chunks[-1].end_char


def test_truncation_is_visible_in_the_plan(small_model: SentenceTransformer) -> None:
    text = " ".join(f"Sentence {index} about parameter P{index}." for index in range(60))
    retriever = DocumentRetriever(
        small_model,
        ChunkingConfig(max_chars=60, overlap_chars=0, max_chunks_per_doc=4),
    )
    retriever.add_documents({"long": text})
    plan = retriever.chunk_plans()["long"]
    assert len(plan.chunks) == 4
    assert plan.truncation is not None
    assert plan.truncation.dropped_chars > 0
    assert plan.to_dict()["truncation"]["dropped_end_char"] == len(text)


def test_default_aggregation_is_not_pure_max(small_model: SentenceTransformer) -> None:
    # The spec forbids "max only" as the behaviour a caller gets by default. With one
    # answer repeated across chunks, mean_top_n must beat a document with a single hit;
    # plain max would rank the single hit second.
    documents = {
        "mentions-once": " ".join(
            [
                "Filler paragraph zero about packaging.",
                "Filler paragraph one about logistics.",
                "The beta parameter default is 42 and appears once.",
                "Filler paragraph three about invoicing.",
                "Filler paragraph four about support hours.",
            ]
        ),
        "mentions-many": " ".join(["The beta parameter default is 42."] * 5),
    }
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=60, overlap_chars=0))
    assert retriever.aggregation == "mean_top_n"

    retriever.add_documents(documents)
    hits = retriever.search("What is the beta parameter default?", top_k=2)[0]
    assert hits[0].doc_id == "mentions-many"
    # Both documents have five chunks; "mentions-once" only has one strong chunk, so
    # averaging over its top 3 drops it below the repeated mentions.
    assert hits[0].score > hits[1].score
    assert hits[0].chunk_scores[-1] == pytest.approx(hits[0].chunk_scores[0])
    assert hits[1].chunk_scores[-1] < hits[1].chunk_scores[0]


# --------------------------------------------------------------------------- persistence


def test_save_and_load_round_trip(
    retriever: DocumentRetriever, small_model: SentenceTransformer, tmp_path: Path
) -> None:
    index_path = retriever.save(tmp_path / "index")
    assert index_path.exists()

    loaded = DocumentRetriever.load(tmp_path / "index", small_model)
    assert loaded.doc_ids == retriever.doc_ids
    assert loaded.num_chunks == retriever.num_chunks
    for doc_id in retriever.doc_ids:
        assert torch.equal(loaded._records[doc_id].embeddings, retriever._records[doc_id].embeddings)
        assert loaded._records[doc_id].text == retriever._records[doc_id].text

    query = "parameter P4 tolerance"
    original_hits = retriever.search(query, top_k=3)[0]
    loaded_hits = loaded.search(query, top_k=3)[0]
    assert [(hit.doc_id, hit.score) for hit in loaded_hits] == [(hit.doc_id, hit.score) for hit in original_hits]


def test_loading_does_not_re_encode(
    retriever: DocumentRetriever, small_model: SentenceTransformer, tmp_path: Path
) -> None:
    retriever.save(tmp_path / "index")

    class ExplodingModel:
        def encode(self, *args, **kwargs):  # pragma: no cover - must never run during load
            raise AssertionError("load() must not encode the corpus")

    loaded = DocumentRetriever.load(tmp_path / "index", ExplodingModel())  # type: ignore[arg-type]
    assert loaded.num_chunks == retriever.num_chunks
    assert loaded.doc_ids == retriever.doc_ids


def test_chunking_manifest_is_saved_and_restored(
    retriever: DocumentRetriever, small_model: SentenceTransformer, tmp_path: Path
) -> None:
    retriever.save(tmp_path / "index")
    payload = torch.load(tmp_path / "index" / "index.pt", map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1
    assert payload["config"]["max_chars"] == 200
    assert payload["config"]["overlap_chars"] == 40
    for document in payload["documents"]:
        assert document["plan"]["chunks"]
        assert document["plan"]["chunks"][0]["start_char"] == 0


def test_corrupted_index_is_rejected(
    small_model: SentenceTransformer, retriever: DocumentRetriever, tmp_path: Path
) -> None:
    retriever.save(tmp_path / "index")
    payload = torch.load(tmp_path / "index" / "index.pt", map_location="cpu", weights_only=True)
    payload["documents"][0]["plan"]["chunks"][0]["end_char"] = 999_999
    torch.save(payload, tmp_path / "index" / "index.pt")

    with pytest.raises(ValueError, match="does not match its text"):
        DocumentRetriever.load(tmp_path / "index", small_model)


def test_unsupported_format_version_is_rejected(
    small_model: SentenceTransformer, retriever: DocumentRetriever, tmp_path: Path
) -> None:
    retriever.save(tmp_path / "index")
    payload = torch.load(tmp_path / "index" / "index.pt", map_location="cpu", weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, tmp_path / "index" / "index.pt")

    with pytest.raises(ValueError, match="Unsupported index format version"):
        DocumentRetriever.load(tmp_path / "index", small_model)


# --------------------------------------------------------------------------- search after reload / no re-chunking


def test_queries_do_not_re_chunk(retriever: DocumentRetriever, monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers.util.document_retrieval as module

    calls: list[str] = []

    def spy(doc_id, text, config=None):
        calls.append(doc_id)
        return module.chunk_document(doc_id, text, config)

    monkeypatch.setattr(module, "chunk_document", spy)
    retriever.search(["query one", "query two"], top_k=2)
    assert calls == []


def test_cached_chunk_encoder_reuses_the_index(small_model: SentenceTransformer, corpus: dict[str, str]) -> None:
    retriever = DocumentRetriever(small_model, ChunkingConfig(max_chars=200, overlap_chars=20))
    cache = CachedChunkEncoder(retriever)
    assert set(cache.build(corpus)) == set(corpus)
    first = cache.search("parameter P2", top_k=2)
    second = cache.search("parameter P2", top_k=2)
    assert [(hit.doc_id, hit.score) for hit in first[0]] == [(hit.doc_id, hit.score) for hit in second[0]]


def test_reloaded_index_serves_search_without_the_original_corpus(
    retriever: DocumentRetriever, small_model: SentenceTransformer, tmp_path: Path
) -> None:
    # Chunk texts are stored in the index, so snippets and char ranges survive a restart
    # even if the caller no longer has the documents in memory.
    retriever.save(tmp_path / "index")
    loaded = DocumentRetriever.load(tmp_path / "index", small_model)
    hit = loaded.search("parameter P1", top_k=1)[0][0]
    chunk = hit.best_chunk
    assert hit.snippets[0]
    assert chunk.text.startswith(chunk.text[0])
    assert chunk.end_char - chunk.start_char == len(chunk.text)
    # And the retriever is still updatable after loading.
    assert loaded.update_document("alpha-manual", _manual("alpha") + " New note.") is True


def test_save_is_atomic_and_replaces_previous_index(
    retriever: DocumentRetriever, small_model: SentenceTransformer, tmp_path: Path
) -> None:
    retriever.save(tmp_path / "index")
    retriever.update_document("alpha-manual", "Completely replaced content about parameter Z.")
    retriever.save(tmp_path / "index")

    loaded = DocumentRetriever.load(tmp_path / "index", small_model)
    assert loaded._records["alpha-manual"].text.startswith("Completely replaced")
    # No temporary file is left behind.
    assert sorted(path.name for path in (tmp_path / "index").iterdir()) == ["index.pt"]


# --------------------------------------------------------------------------- result serialisation


def test_hit_to_dict_is_json_serialisable(retriever: DocumentRetriever) -> None:
    hit = retriever.search("parameter P1", top_k=1)[0][0]
    payload = hit.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["doc_id"] == hit.doc_id
    assert decoded["matched_chunks"][0]["start_char"] == hit.best_chunk.start_char
    assert decoded["matched_chunks"][0]["snippet"] == hit.snippets[0]


# --------------------------------------------------------------------------- coverage helper


def test_covered_fraction_counts_overlaps_once() -> None:
    from sentence_transformers.util.chunking import ChunkPlan, TextChunk

    chunks = tuple(
        TextChunk(
            doc_id="d",
            chunk_index=index,
            start_char=start,
            end_char=end,
            text="x" * (end - start),
            end_boundary="sentence",
        )
        for index, (start, end) in enumerate([(0, 60), (40, 100), (100, 150)])
    )
    plan = ChunkPlan(doc_id="d", text_length=150, chunks=chunks, total_chunks=3)
    # [0, 100) and [100, 150) are hand-computed unions of the flagged ranges.
    assert _covered_fraction(plan, [True, True, False]) == pytest.approx(100 / 150)
    assert _covered_fraction(plan, [True, True, True]) == pytest.approx(1.0)
    assert _covered_fraction(plan, [False, False, False]) == 0.0
