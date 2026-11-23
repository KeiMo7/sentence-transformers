from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers.util.chunking import (
    ChunkingConfig,
    ChunkingError,
    chunk_document,
    chunk_text,
)


def _english_doc(sentences: int = 40) -> str:
    # Each sentence is a distinct numbered fact, so a chunk boundary can be checked
    # against the sentence it ends on, and content overlap can be measured exactly.
    return "".join(f"Sentence {index} describes parameter P{index} in detail. " for index in range(sentences))


def _cjk_doc(sentences: int = 40) -> str:
    return "".join(f"第{index}句说明参数P{index}的取值。" for index in range(sentences))


def test_chunks_are_contiguous_and_traceable() -> None:
    text = _cjk_doc()
    plan = chunk_document("manual", text, ChunkingConfig(max_chars=80, overlap_chars=20))

    assert plan.total_chunks == len(plan.chunks)
    assert plan.chunks[0].start_char == 0
    assert plan.chunks[-1].end_char == len(text)
    for chunk in plan.chunks:
        assert chunk.doc_id == "manual"
        assert chunk.text == text[chunk.start_char : chunk.end_char]
        assert 0 < len(chunk.text) <= 80
    for previous, current in zip(plan.chunks, plan.chunks[1:]):
        assert current.start_char < previous.end_char, "consecutive chunks must overlap"
        assert current.start_char >= previous.start_char


def test_cut_points_never_land_inside_a_sentence() -> None:
    """Chunk ends must coincide with sentence ends, never with a sentence's interior."""
    text = _cjk_doc()
    config = ChunkingConfig(max_chars=80, overlap_chars=20)
    plan = chunk_document("manual", text, config)

    sentence_ends = set()
    cursor = 0
    for index in range(40):
        cursor += len(f"第{index}句说明参数P{index}的取值。")
        sentence_ends.add(cursor)

    for chunk in plan.chunks[:-1]:
        assert chunk.end_char in sentence_ends, (
            f"chunk {chunk.chunk_index} ends at {chunk.end_char} inside a sentence: {chunk.text[-12:]!r}"
        )
        assert chunk.end_boundary == "sentence"
        assert not chunk.forced


def test_cut_points_never_land_inside_an_english_sentence() -> None:
    text = _english_doc()
    plan = chunk_document("manual", text, ChunkingConfig(max_chars=120, overlap_chars=30))

    # Every chunk ends right before the next sentence starts (after its trailing space).
    for chunk in plan.chunks[:-1]:
        next_start = chunk.end_char
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        assert text[next_start:].startswith("Sentence "), chunk.text[-20:]


def test_overlap_preserves_context_across_boundaries() -> None:
    text = _english_doc()
    config = ChunkingConfig(max_chars=120, overlap_chars=40)
    plan = chunk_document("manual", text, config)

    for previous, current in zip(plan.chunks, plan.chunks[1:]):
        overlap = text[current.start_char : previous.end_char]
        assert 0 < len(overlap) <= 120
        # The overlap is whole sentences, so it starts at a sentence start.
        assert overlap.lstrip().startswith("Sentence ")


def test_chunking_is_deterministic() -> None:
    text = _cjk_doc() + _english_doc()
    config = ChunkingConfig(max_chars=150, overlap_chars=30)

    first = chunk_document("manual", text, config)
    second = chunk_document("manual", text, config)

    assert [chunk.to_dict() for chunk in first.chunks] == [chunk.to_dict() for chunk in second.chunks]
    assert [chunk.text for chunk in first.chunks] == [chunk.text for chunk in second.chunks]


def test_short_document_is_not_split() -> None:
    text = "A short note that fits in one chunk."
    plan = chunk_document("note", text, ChunkingConfig(max_chars=1000))

    assert plan.total_chunks == 1
    assert plan.chunks[0].text == text
    assert plan.chunks[0].start_char == 0
    assert plan.chunks[0].end_char == len(text)
    assert plan.chunks[0].end_boundary == "end"
    assert plan.truncation is None
    assert plan.forced_chunks == 0


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t "])
def test_empty_document_yields_no_chunk(text: str) -> None:
    plan = chunk_document("empty", text, ChunkingConfig())
    assert plan.is_empty
    assert plan.chunks == ()
    assert plan.total_chunks == 0


def test_oversized_sentence_splits_at_clause_then_word() -> None:
    # One sentence with no sentence terminator but plenty of clause marks.
    sentence = "，" + "参数取值说明" * 20
    text = f"开头{sentence}。结尾句。"
    plan = chunk_document("long", text, ChunkingConfig(max_chars=60, overlap_chars=10))

    assert plan.total_chunks > 1
    assert any(chunk.end_boundary == "clause" for chunk in plan.chunks)
    for chunk in plan.chunks:
        assert len(chunk.text) <= 60
        assert chunk.text == text[chunk.start_char : chunk.end_char]


def test_hard_split_is_reported_and_can_be_disabled() -> None:
    text = "x" * 500
    plan = chunk_document("blob", text, ChunkingConfig(max_chars=100, overlap_chars=10))

    assert plan.forced_chunks == plan.total_chunks - 1
    assert all(chunk.forced for chunk in plan.chunks[:-1])
    assert all(len(chunk.text) <= 100 for chunk in plan.chunks)

    with pytest.raises(ChunkingError) as excinfo:
        chunk_document("blob", text, ChunkingConfig(max_chars=100, overlap_chars=10, allow_hard_split=False))
    assert "'blob'" in str(excinfo.value)
    assert excinfo.value.doc_id == "blob"
    assert excinfo.value.position == 0


def test_truncation_reports_the_dropped_range(caplog: pytest.LogCaptureFixture) -> None:
    text = _cjk_doc(sentences=100)
    config = ChunkingConfig(max_chars=40, overlap_chars=10, max_chunks_per_doc=5)
    with caplog.at_level("WARNING", logger="sentence_transformers.util.chunking"):
        plan = chunk_document("manual", text, config)

    assert len(plan.chunks) == 5
    assert plan.total_chunks > 5
    assert plan.truncation is not None
    assert plan.truncation.total_chunks == plan.total_chunks
    assert plan.truncation.kept_chunks == 5
    assert plan.truncation.dropped_start_char == plan.chunks[-1].end_char
    assert plan.truncation.dropped_end_char == len(text)
    assert plan.truncation.dropped_chars > 0
    assert "max_chunks_per_doc" in caplog.text
    assert "manual" in caplog.text
    assert "cannot be retrieved" in caplog.text


def test_coverage_is_complete_without_truncation() -> None:
    text = _cjk_doc()
    plan = chunk_document("manual", text, ChunkingConfig(max_chars=80, overlap_chars=20))

    covered: set[int] = set()
    for chunk in plan.chunks:
        covered.update(range(chunk.start_char, chunk.end_char))
    assert len(covered) == len(text)


def test_chunk_text_helper() -> None:
    chunks = chunk_text("第一句。第二句。", ChunkingConfig(max_chars=10, overlap_chars=0))
    assert isinstance(chunks, list)
    assert [chunk.text for chunk in chunks] == ["第一句。第二句。"]  # fits in one chunk
    assert chunk_text("") == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 0},
        {"max_chars": 10, "overlap_chars": 10},
        {"max_chars": 10, "overlap_chars": -1},
        {"max_chunks_per_doc": 0},
        {"sentence_terminators": ""},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError):
        chunk_document("doc", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        chunk_document(None, "text")  # type: ignore[arg-type]


def test_chunk_plan_serialisation_round_trips() -> None:
    text = _cjk_doc(10)
    plan = chunk_document("manual", text, ChunkingConfig(max_chars=60, overlap_chars=10))
    payload = plan.to_dict()

    assert payload["doc_id"] == "manual"
    assert payload["text_length"] == len(text)
    assert len(payload["chunks"]) == len(plan.chunks)
    assert payload["chunks"][0]["start_char"] == 0
    assert "text" not in payload["chunks"][0]


def test_chunk_boundaries_are_stable_when_whitespace_changes() -> None:
    base = _cjk_doc(20)
    with_spaces = base.replace("。", "。\n")
    config = ChunkingConfig(max_chars=80, overlap_chars=15)

    first = chunk_document("a", base, config)
    second = chunk_document("b", with_spaces, config)
    assert len(first.chunks) == len(second.chunks)

    # Adding line breaks does not move any cut into a sentence.
    for chunk in second.chunks[:-1]:
        assert chunk.end_boundary in {"sentence", "clause", "word"}


def test_numpy_independence_of_chunking() -> None:
    # Chunking must not depend on any model or tensor library state.
    plan = chunk_document("doc", "One. Two. Three.", ChunkingConfig(max_chars=5, overlap_chars=0))
    assert all(isinstance(chunk.start_char, int) for chunk in plan.chunks)
    assert np.asarray([chunk.end_char for chunk in plan.chunks]).dtype.kind == "i"
