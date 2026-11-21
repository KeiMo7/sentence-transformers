"""Tests for the resumable, sharded :class:`~sentence_transformers.util.EmbeddingStore`."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util import (
    CorpusPlan,
    EmbeddingSpec,
    EmbeddingStore,
    EmbeddingStoreError,
    FingerprintMismatchError,
    PlanMismatchError,
    ShardClaimedError,
    StoreCorruptionError,
    build_plan,
)


@pytest.fixture(scope="module")
def tiny_model() -> SentenceTransformer:
    """A tiny deterministic model built locally, so these tests never touch the Hub."""
    vocab = ["hello", "world", "a", "b", "cat", "dog", "this", "is", "long", "text", "the", "model", "vector"]
    embeddings = torch.rand(len(vocab), 24, generator=torch.Generator().manual_seed(7))
    module = WordEmbeddings(tokenizer=WhitespaceTokenizer(vocab=vocab), embedding_weights=embeddings)
    model = SentenceTransformer(modules=[module, Pooling(24, "mean")], device="cpu")
    model.model_card_data.generate_widget_examples = False
    return model


def _texts(count: int) -> list[str]:
    return [f"hello world {index}" for index in range(count)]


def _reference(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=4, convert_to_numpy=True, show_progress_bar=False)


# -------------------------------------------------------------------------------------- corpus shapes


@pytest.mark.parametrize(
    ("ids", "normalized"),
    [
        ([], "none"),
        ([], "nfc"),
        (["a/b 中文 id", "a/b 中文 id", "", "  ", "\x00control\x07chars"], "none"),
        ([f"id-{index}" for index in range(50)], "nfc-strip-control"),
    ],
)
def test_build_plan_deduplicates_in_first_seen_order(tmp_path: Path, ids: list[str], normalized: str) -> None:
    corpus = [(row_id, f"hello world {index}") for index, row_id in enumerate(ids)]
    plan = build_plan(corpus, text_normalization=normalized)
    assert plan.ids == tuple(dict.fromkeys(ids))
    assert plan.num_duplicates_dropped == len(ids) - len(plan.ids)
    assert len(plan.text_hashes) == len(plan)


def test_run_from_memory_list_matches_encode(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    texts = _texts(23)
    store = EmbeddingStore(tmp_path / "store", shard_size=5)
    stats = store.run(tiny_model, texts, batch_size=4)
    assert stats.num_rows == 23
    assert stats.num_shards_total == 5
    assert stats.num_shards_written == 5

    ids, vectors = store.load()
    assert ids == [str(index) for index in range(23)]
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))


def test_run_from_text_file_uses_line_identifiers(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"hello world\nhello world 2\n\xff\xfe broken bytes\n")
    store = EmbeddingStore(tmp_path / "store", shard_size=2)
    store.run(tiny_model, path, batch_size=2)

    ids, vectors = store.load()
    assert ids == ["line:0", "line:1", "line:2"]
    # Undecodable bytes are replaced rather than raising.
    expected = _reference(tiny_model, ["hello world", "hello world 2", "�� broken bytes"])
    np.testing.assert_array_equal(vectors, expected)


def test_run_from_mapping_and_pairs(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    corpus = {"/slash path/商品 1": "hello world", "商品 1": "cat dog"}
    store = EmbeddingStore(tmp_path / "mapping", shard_size=1)
    store.run(tiny_model, corpus)
    assert store.get_ids() == ["/slash path/商品 1", "商品 1"]

    pairs = [("a", "hello world"), ("b", "cat dog")]
    store = EmbeddingStore(tmp_path / "pairs", shard_size=1)
    store.run(tiny_model, pairs)
    assert store.get_ids() == ["a", "b"]


def test_empty_corpus_produces_a_verified_empty_store(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store")
    stats = store.run(tiny_model, [], batch_size=4)
    assert stats.num_rows == 0
    assert stats.num_shards_total == 0
    assert store.verify(model=tiny_model).ok
    ids, vectors = store.load()
    assert ids == []
    assert vectors.shape[0] == 0


def test_empty_store_has_no_leftover_work_files(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store", shard_size=4)
    store.run(tiny_model, _texts(9))
    assert list(store.path.glob(".corpus-*")) == []
    assert not any(".tmp-" in path.name for path in store.path.iterdir())


def test_file_corpus_is_read_line_by_line(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "corpus.txt"
    lines = [f"hello world {index}" for index in range(40)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = EmbeddingStore(tmp_path / "store", shard_size=7)
    stats = store.run(tiny_model, path, batch_size=4)
    assert stats.num_rows == 40
    assert store.get_ids() == [f"line:{index}" for index in range(40)]
    _, vectors = store.load()
    np.testing.assert_array_equal(vectors, _reference(tiny_model, lines))


def test_duplicate_identifiers_are_kept_once(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    corpus = [("dup", "hello world"), ("other", "cat dog"), ("dup", "a different text")]
    store = EmbeddingStore(tmp_path / "store")
    stats = store.run(tiny_model, corpus)
    assert stats.num_duplicates_dropped == 1
    ids, vectors = store.load()
    assert ids == ["dup", "other"]
    np.testing.assert_array_equal(vectors[0], _reference(tiny_model, ["hello world"])[0])


def test_long_single_text(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    text = "hello world " * 5000
    store = EmbeddingStore(tmp_path / "store", shard_size=10)
    store.run(tiny_model, [("long", text)])
    ids, vectors = store.load()
    assert ids == ["long"]
    np.testing.assert_array_equal(vectors, _reference(tiny_model, [text]))


# -------------------------------------------------------------------------------------- resume


def test_resume_only_encodes_missing_shards(tmp_path: Path, tiny_model: SentenceTransformer, monkeypatch) -> None:
    texts = _texts(25)
    path = tmp_path / "store"

    # Simulate a kill after three of five shards.
    EmbeddingStore(path, shard_size=5).run(tiny_model, texts, batch_size=4, shard_ids=[0, 1, 2])
    partial = EmbeddingStore(path)
    assert partial.pending_shards() == [3, 4]
    assert not partial.verify().ok

    calls: list[list[str]] = []
    original_encode = tiny_model.encode

    def recording_encode(*args, **kwargs):
        if args and isinstance(args[0], list):
            calls.append(list(args[0]))
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "encode", recording_encode)
    stats = EmbeddingStore(path).run(tiny_model, texts, batch_size=4)
    assert stats.num_shards_written == 2
    assert stats.num_shards_reused == 3
    assert stats.num_encoded == 10

    encoded_rows = [row for call in calls for row in call]
    # Only the two missing shards were re-encoded.
    assert encoded_rows == texts[15:25]

    ids, vectors = EmbeddingStore(path).load()
    assert ids == [str(index) for index in range(25)]
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))
    assert EmbeddingStore(path).verify(model=tiny_model).ok


def test_resume_after_manifest_lost_shard_entry_repairs_it(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    texts = _texts(12)
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, texts)

    # Simulate a crash between writing the last shard and recording it.
    store = EmbeddingStore(path)
    manifest = json.loads(store.manifest_path.read_text())
    removed = manifest["shards"].pop()
    store.manifest_path.write_text(json.dumps(manifest))
    assert removed["shard"] == 2
    assert EmbeddingStore(path).pending_shards() == [2]

    stats = EmbeddingStore(path).run(tiny_model, texts)
    assert stats.num_shards_written == 1
    ids, vectors = EmbeddingStore(path).load()
    assert ids == [str(index) for index in range(12)]
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))


def test_disk_full_mid_shard_leaves_the_shard_incomplete(
    tmp_path: Path, tiny_model: SentenceTransformer, monkeypatch
) -> None:
    path = tmp_path / "store"
    texts = _texts(20)
    store = EmbeddingStore(path, shard_size=5)
    original_write = store._write_shard
    calls = {"count": 0}

    def failing_write(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(store, "_write_shard", failing_write)
    with pytest.raises(OSError, match="No space left"):
        store.run(tiny_model, texts, batch_size=4)

    partial = EmbeddingStore(path)
    assert sorted(partial.completed_shards()) == [0, 1]
    assert partial.pending_shards() == [2, 3]
    assert not partial.verify().ok
    # A partially written shard must never be loadable as if it were complete.
    with pytest.raises(EmbeddingStoreError, match="incomplete"):
        partial.load()

    partial.run(tiny_model, texts, batch_size=4)
    _, vectors = EmbeddingStore(path).load()
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))
    assert EmbeddingStore(path).verify(model=tiny_model).ok


def test_truncated_shard_is_not_treated_as_complete(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    texts = _texts(12)
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, texts)

    store = EmbeddingStore(path)
    entry = store._sorted_entries()[1]
    data_path = store.path / entry["data"]
    data_path.write_bytes(data_path.read_bytes()[:64])

    assert EmbeddingStore(path).pending_shards() == [1]
    assert not EmbeddingStore(path).verify().ok

    stats = EmbeddingStore(path).run(tiny_model, texts)
    assert stats.num_shards_written == 1
    _, vectors = EmbeddingStore(path).load()
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))


def test_empty_shard_file_is_not_treated_as_complete(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    texts = _texts(8)
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, texts)

    store = EmbeddingStore(path)
    entry = store._sorted_entries()[0]
    (store.path / entry["data"]).write_bytes(b"")

    assert 0 in EmbeddingStore(path).pending_shards()
    EmbeddingStore(path).run(tiny_model, texts)
    _, vectors = EmbeddingStore(path).load()
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))


def test_resume_with_a_different_corpus_is_refused(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, [("a", "hello"), ("b", "world")])

    with pytest.raises(PlanMismatchError, match="different corpus") as excinfo:
        EmbeddingStore(path).run(tiny_model, [("a", "hello"), ("c", "world")])
    assert any("identifiers" in difference for difference in excinfo.value.differences)

    with pytest.raises(PlanMismatchError, match="row count"):
        EmbeddingStore(path).run(tiny_model, [("a", "hello")])


def test_resume_with_changed_text_points_at_apply_delta(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, [("a", "hello"), ("b", "world")])
    with pytest.raises(PlanMismatchError, match="apply_delta") as excinfo:
        EmbeddingStore(path).run(tiny_model, [("a", "hello"), ("b", "cat")])
    assert any("texts changed" in difference for difference in excinfo.value.differences)


def test_shard_ids_out_of_range(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store", shard_size=4)
    with pytest.raises(ValueError, match="outside the store's"):
        store.run(tiny_model, _texts(4), shard_ids=[1])


# -------------------------------------------------------------------------------------- fingerprint


def test_fingerprint_changes_with_every_pinned_setting(tiny_model: SentenceTransformer) -> None:
    base = EmbeddingSpec.from_model(tiny_model)
    assert EmbeddingSpec.from_model(tiny_model).fingerprint == base.fingerprint
    assert EmbeddingSpec.from_model(tiny_model, normalize_embeddings=True).fingerprint != base.fingerprint
    assert EmbeddingSpec.from_model(tiny_model, truncate_dim=8).fingerprint != base.fingerprint
    assert EmbeddingSpec.from_model(tiny_model, precision="int8").fingerprint != base.fingerprint
    assert EmbeddingSpec.from_model(tiny_model, prompt="query: ").fingerprint != base.fingerprint
    assert EmbeddingSpec.from_model(tiny_model, task="query").fingerprint != base.fingerprint
    assert EmbeddingSpec.from_model(tiny_model, text_normalization="nfc").fingerprint != base.fingerprint


def test_fingerprint_detects_a_changed_model(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, _texts(6))

    # A model with the same architecture but different weights must not be mixed in.
    other = SentenceTransformer(
        modules=[
            WordEmbeddings(
                tokenizer=WhitespaceTokenizer(vocab=["hello", "world"]),
                embedding_weights=torch.rand(2, 24, generator=torch.Generator().manual_seed(99)),
            ),
            Pooling(24, "mean"),
        ],
        device="cpu",
    )
    with pytest.raises(FingerprintMismatchError) as excinfo:
        EmbeddingStore(path).run(other, _texts(6))
    assert any(difference.startswith("model_digest:") for difference in excinfo.value.differences)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"normalize_embeddings": True}, "normalize_embeddings:"),
        ({"truncate_dim": 8}, "truncate_dim:"),
        ({"precision": "int8"}, "precision:"),
    ],
)
def test_resume_refuses_mismatched_settings(
    tmp_path: Path, tiny_model: SentenceTransformer, kwargs: dict, field: str
) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, _texts(6))
    with pytest.raises(FingerprintMismatchError) as excinfo:
        EmbeddingStore(path).run(tiny_model, _texts(6), **kwargs)
    assert any(difference.startswith(field) for difference in excinfo.value.differences)
    assert field.rstrip(":") in str(excinfo.value)


def test_cannot_reuse_a_store_for_a_different_template(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, _texts(6))
    with pytest.raises(FingerprintMismatchError) as excinfo:
        EmbeddingStore(path).run(tiny_model, _texts(6), prompt="query: ")
    assert any(difference.startswith("template:") for difference in excinfo.value.differences)


def test_same_fingerprint_is_recorded_in_the_manifest(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store")
    store.run(tiny_model, _texts(4), normalize_embeddings=True)
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["fingerprint"] == EmbeddingSpec.from_model(tiny_model, normalize_embeddings=True).fingerprint
    assert manifest["spec"]["normalize_embeddings"] is True
    assert manifest["spec"]["model"] == "<in-memory:SentenceTransformer>"


def test_verify_reports_the_differing_fingerprint_field(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, _texts(4))
    report = EmbeddingStore(path).verify(spec=EmbeddingSpec.from_model(tiny_model, normalize_embeddings=True))
    assert not report.ok
    assert any(difference.startswith("normalize_embeddings:") for difference in report.differences)


# -------------------------------------------------------------------------------------- incremental updates


def test_delta_adds_modifies_and_deletes(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    original = [(f"id/{index}", f"hello world {index}") for index in range(20)]
    EmbeddingStore(path, shard_size=6).run(tiny_model, original)

    stats = EmbeddingStore(path).apply_delta(
        tiny_model,
        upserts=[("new/1", "cat dog"), ("new/2", "a b"), ("id/3", "the model vector")],
        deletes=["id/5", "id/7"],
    )
    assert (stats.num_added, stats.num_modified, stats.num_deleted) == (2, 1, 2)
    assert stats.num_rows == 20

    expected_ids = [row_id for row_id, _ in original if row_id not in {"id/5", "id/7"}] + ["new/1", "new/2"]
    expected_texts = [
        "the model vector" if row_id == "id/3" else text for row_id, text in original if row_id not in {"id/5", "id/7"}
    ] + ["cat dog", "a b"]

    store = EmbeddingStore(path)
    ids, vectors = store.load()
    assert ids == expected_ids
    np.testing.assert_array_equal(vectors, _reference(tiny_model, expected_texts))
    assert store.verify(model=tiny_model).ok


def test_delta_only_reencodes_affected_rows(tmp_path: Path, tiny_model: SentenceTransformer, monkeypatch) -> None:
    path = tmp_path / "store"
    original = [(f"id-{index}", f"hello world {index}") for index in range(20)]
    EmbeddingStore(path, shard_size=5).run(tiny_model, original)

    encoded: list[str] = []
    original_encode = tiny_model.encode

    def recording_encode(inputs, *args, **kwargs):
        if isinstance(inputs, list):
            encoded.extend(inputs)
        return original_encode(inputs, *args, **kwargs)

    monkeypatch.setattr(tiny_model, "encode", recording_encode)
    EmbeddingStore(path).apply_delta(
        tiny_model,
        upserts=[("id-4", "cat dog"), ("brand-new", "a b")],
        deletes=["id-2"],
    )
    assert encoded == ["cat dog", "a b"]

    # Re-applying an identical upsert must not re-encode anything.
    encoded.clear()
    stats = EmbeddingStore(path).apply_delta(tiny_model, upserts=[("id-4", "cat dog")])
    assert encoded == []
    assert stats.num_unchanged == 1


def test_delta_matches_a_one_shot_rebuild(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    original = [(f"id-{index}", f"hello world {index}") for index in range(17)]
    delta_path = tmp_path / "delta"
    EmbeddingStore(delta_path, shard_size=4).run(tiny_model, original)
    EmbeddingStore(delta_path).apply_delta(
        tiny_model,
        upserts=[("id-1", "the model vector"), ("id-16", "a b"), ("new/0", "cat dog")],
        deletes=["id-0", "id-9"],
    )

    expected = [
        (row_id, "the model vector" if row_id == "id-1" else text)
        for row_id, text in original
        if row_id not in {"id-0", "id-9"}
    ]
    expected[-1] = ("id-16", "a b")
    expected += [("new/0", "cat dog")]
    one_shot_path = tmp_path / "one-shot"
    EmbeddingStore(one_shot_path, shard_size=4).run(tiny_model, expected)

    delta_ids, delta_vectors = EmbeddingStore(delta_path).load()
    one_shot_ids, one_shot_vectors = EmbeddingStore(one_shot_path).load()
    assert delta_ids == one_shot_ids
    np.testing.assert_array_equal(delta_vectors, one_shot_vectors)


def test_delta_rejects_ambiguous_input(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, [("a", "hello"), ("b", "world")], batch_size=2)
    store = EmbeddingStore(path)

    with pytest.raises(ValueError, match="not stored"):
        store.apply_delta(tiny_model, deletes=["missing"])
    with pytest.raises(ValueError, match="both deleted and upserted"):
        store.apply_delta(tiny_model, upserts=[("a", "hello")], deletes=["a"])
    with pytest.raises(ValueError, match="more than once"):
        store.apply_delta(tiny_model, upserts=[("a", "hello"), ("a", "world")])
    EmbeddingStore(tmp_path / "empty").run(tiny_model, [])
    with pytest.raises(EmbeddingStoreError, match="still empty"):
        EmbeddingStore(tmp_path / "empty").apply_delta(tiny_model, upserts=[("a", "hello")])


def test_delta_fingerprint_mismatch_is_refused(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, [("a", "hello"), ("b", "world")])
    with pytest.raises(FingerprintMismatchError):
        EmbeddingStore(path).apply_delta(tiny_model, upserts=[("c", "cat")], normalize_embeddings=True)


def test_delta_after_partial_store_is_refused(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, _texts(12), shard_ids=[0])
    with pytest.raises(EmbeddingStoreError, match="still incomplete"):
        EmbeddingStore(path).apply_delta(tiny_model, upserts=[("new", "hello")])


# -------------------------------------------------------------------------------------- concurrency


def test_two_workers_claiming_the_same_shard_do_not_double_write(
    tmp_path: Path, tiny_model: SentenceTransformer
) -> None:
    path = tmp_path / "store"
    store = EmbeddingStore(path, shard_size=4)
    store.run(tiny_model, _texts(16), batch_size=2, shard_ids=[0, 1])

    other = EmbeddingStore(path)
    with other._advisory_lock(other._claim_path(2), blocking=False) as acquired:
        assert acquired
        with pytest.raises(ShardClaimedError, match="currently being written"):
            store.run(tiny_model, _texts(16), shard_ids=[2])

    # Once the claim is released, the missing shard is encoded normally.
    stats = EmbeddingStore(path).run(tiny_model, _texts(16), shard_ids=[2, 3])
    assert stats.num_shards_written == 2
    ids, _ = EmbeddingStore(path).load()
    assert ids == [str(index) for index in range(16)]


def test_two_workers_split_disjoint_shards(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    texts = _texts(30)
    EmbeddingStore(path, shard_size=5).run(tiny_model, texts, shard_ids=[0, 1, 2])
    EmbeddingStore(path).run(tiny_model, texts, shard_ids=[3, 4, 5])

    ids, vectors = EmbeddingStore(path).load()
    assert ids == [str(index) for index in range(30)]
    assert len(set(ids)) == 30
    np.testing.assert_array_equal(vectors, _reference(tiny_model, texts))


def test_concurrent_runs_produce_one_consistent_store(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    """Both racing jobs see a complete store and neither produces duplicate rows."""
    path = tmp_path / "store"
    texts = _texts(16)
    first = EmbeddingStore(path, shard_size=4)
    second = EmbeddingStore(path, shard_size=4)
    first.run(tiny_model, texts, shard_ids=[0, 1])
    second.run(tiny_model, texts, shard_ids=[2, 3])
    # Each job resumes and finds nothing left to do.
    assert first.run(tiny_model, texts).num_shards_written == 0
    assert second.run(tiny_model, texts).num_shards_written == 0

    ids, _ = EmbeddingStore(path).load()
    assert ids == [str(index) for index in range(16)]
    assert len(ids) == len(set(ids))


def test_store_lock_timeout(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    store = EmbeddingStore(path)
    store.run(tiny_model, _texts(2))
    with store._store_lock() as acquired:
        assert acquired
    # After release the lock is available again.
    with EmbeddingStore(path)._store_lock(timeout=0.1) as acquired:
        assert acquired


# -------------------------------------------------------------------------------------- verification


def test_verify_is_detailed_and_does_not_encode(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, _texts(10), shard_ids=[0])
    report = EmbeddingStore(path).verify(model=tiny_model)
    assert not report.ok
    assert report.num_rows == 10
    assert report.num_shards == 1
    assert report.dim == 24
    assert any("shards are missing" in difference for difference in report.differences)
    assert "INCOMPLETE" in str(report)

    report = EmbeddingStore(tmp_path / "empty").verify()
    assert not report.ok
    assert any("no manifest" in difference for difference in report.differences)


def test_verify_detects_corrupt_bytes(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path).run(tiny_model, _texts(8), batch_size=4)
    store = EmbeddingStore(path)
    entry = store._sorted_entries()[0]
    data_path = store.path / entry["data"]
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 0xFF  # keep the length, break the digest
    data_path.write_bytes(bytes(payload))

    report = store.verify()
    assert not report.ok
    assert any("SHA-256" in problem for shard in report.shards for problem in shard.problems)
    with pytest.raises(StoreCorruptionError, match="SHA-256"):
        store.load(verify_checksums=True)


def test_verify_detects_a_duplicated_identifier(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, _texts(8))
    store = EmbeddingStore(path)
    manifest = json.loads(store.manifest_path.read_text())
    first_meta_path = store.path / manifest["shards"][0]["meta"]
    first_meta = json.loads(first_meta_path.read_text())
    second_meta_path = store.path / manifest["shards"][1]["meta"]
    second_meta = json.loads(second_meta_path.read_text())
    second_meta["ids"] = first_meta["ids"][:1] + second_meta["ids"][1:]
    second_meta_path.write_text(json.dumps(second_meta))

    # Force the manifest to trust the edited metadata.
    manifest["shards"][1]["ids_sha256"] = (
        __import__("hashlib").sha256(("\n".join(second_meta["ids"]) + "\n").encode()).hexdigest()
    )
    store.manifest_path.write_text(json.dumps(manifest))

    report = EmbeddingStore(path).verify()
    assert any("more than one row" in difference for difference in report.differences)


def test_verify_fingerprint_matching_store_passes(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store", shard_size=3)
    store.run(tiny_model, _texts(9), normalize_embeddings=True, truncate_dim=12)
    report = store.verify(model=tiny_model)
    assert report.ok, str(report)
    assert report.num_shards == 3
    assert report.dim == 12


def test_identical_vectors_for_different_rows_do_not_share_a_shard_file(
    tmp_path: Path, tiny_model: SentenceTransformer
) -> None:
    """Two shard generations can hold byte-identical vectors for different identifiers.

    Their files must stay distinct, or a later write silently replaces the vectors of an earlier generation.
    """
    path = tmp_path / "store"
    # Both texts encode to the exact same vector, so the two shards' .npy bytes are identical.
    EmbeddingStore(path, shard_size=2).run(tiny_model, [("a", "hello"), ("b", "hello"), ("c", "world")])
    EmbeddingStore(path).apply_delta(tiny_model, upserts=[("d", "hello")], deletes=["a"])

    store = EmbeddingStore(path)
    report = store.verify(model=tiny_model)
    assert report.ok, str(report)
    ids, vectors = store.load()
    assert ids == ["b", "c", "d"]
    np.testing.assert_array_equal(vectors, _reference(tiny_model, ["hello", "world", "hello"]))


# -------------------------------------------------------------------------------------- compatibility


def test_short_and_singular_encoding_is_unchanged(
    tiny_model: SentenceTransformer,
) -> None:
    """Storing must not alter what plain encode returns for the shapes the library already supports."""
    single = tiny_model.encode("hello world")
    batch = tiny_model.encode(["hello world", "cat dog"])
    empty = tiny_model.encode([])
    assert single.shape == (24,)
    assert batch.shape == (2, 24)
    assert empty.shape[0] == 0
    np.testing.assert_array_equal(batch[0], single)


def test_length_sorted_batching_order_is_preserved(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    """encode() sorts by length and restores order; shards must keep that final order exactly."""
    texts = [
        "hello",
        "hello world this is a long text vector model",
        "cat",
        "a b a b a b a b a b a b",
        "world",
        "the model vector is long text a b",
    ]
    store = EmbeddingStore(tmp_path / "store", shard_size=2)
    store.run(tiny_model, texts, batch_size=4)
    ids, vectors = store.load()
    assert ids == [str(index) for index in range(len(texts))]
    # Per-row encoding equals the row at its input position.
    for index, text in enumerate(texts):
        np.testing.assert_array_equal(vectors[index], tiny_model.encode(text))


def test_prune_removes_unreferenced_files(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    path = tmp_path / "store"
    EmbeddingStore(path, shard_size=4).run(tiny_model, _texts(8))
    store = EmbeddingStore(path)
    orphan = store.shards_path / "shard-09999-deadbeefdeadbeef.npy"
    orphan.write_bytes(b"orphan")
    (store.shards_path / "shard-09999-deadbeefdeadbeef.json").write_text("{}")
    (store.path / ".manifest.json.tmp-1-abcd").write_text("tmp")

    # Files younger than the grace period are kept so a concurrent writer's fresh shards are never pruned.
    assert store.prune() == []
    assert orphan.exists()

    removed = store.prune(min_age_seconds=0)
    assert "shard-09999-deadbeefdeadbeef.npy" in removed
    assert not orphan.exists()


def test_manifest_is_self_describing(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store", shard_size=4)
    store.run(tiny_model, _texts(10), normalize_embeddings=True)
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["num_rows"] == 10
    assert manifest["num_shards"] == 3
    assert manifest["dim"] == 24
    assert manifest["spec"]["normalize_embeddings"] is True
    assert manifest["plan_hash"] and manifest["ids_hash"]
    assert [entry["first_row"] for entry in manifest["shards"]] == [0, 4, 8]
    assert [entry["num_rows"] for entry in manifest["shards"]] == [4, 4, 2]
    assert all(entry["data_sha256"] for entry in manifest["shards"])


def test_corpus_plan_differences_are_informative() -> None:
    first = CorpusPlan(("a", "b"), ("h1", "h2"))
    second = CorpusPlan(("a", "b"), ("h1", "changed"))
    differences = first.differences(second)
    assert any("texts changed" in difference for difference in differences)
    assert CorpusPlan(("a",), ("h",)).differences(CorpusPlan(("a", "b"), ("h", "h2")))


def test_create_with_explicit_spec_then_run_empty(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    spec = EmbeddingSpec.from_model(tiny_model, normalize_embeddings=True)
    store = EmbeddingStore(tmp_path / "store", spec=spec)
    assert store.num_rows == 0
    assert store.fingerprint == spec.fingerprint
    assert store.verify(model=tiny_model, spec=spec).ok
    store.run(tiny_model, [])
    assert store.verify().ok


def test_build_plan_never_encodes(tmp_path: Path) -> None:
    rows = [("id/中文 名称", "hello"), ("id/中文 名称", "ignored"), ("other", "world")]
    plan = build_plan(rows)
    assert plan.ids == ("id/中文 名称", "other")
    assert plan.num_duplicates_dropped == 1
    assert plan.ids_hash == CorpusPlan(("id/中文 名称", "other"), (None, None)).ids_hash


def test_verify_detects_a_tampered_plan_file(tmp_path: Path, tiny_model: SentenceTransformer) -> None:
    store = EmbeddingStore(tmp_path / "store", shard_size=4)
    store.run(tiny_model, _texts(8))
    plan_path = store.path / store.manifest["plan_file"]
    plan = json.loads(plan_path.read_text())
    plan["ids"][0] = "tampered"
    plan_path.write_text(json.dumps(plan))

    report = store.verify()
    assert not report.ok
    assert any("plan" in difference for difference in report.differences)
