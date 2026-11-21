"""Resumable, sharded and incrementally updatable embedding stores.

An :class:`EmbeddingStore` writes sentence embeddings to a directory of immutable shard files plus a
single ``manifest.json``. Every shard is written to a uniquely named file, flushed to disk and only then
recorded in the manifest, so a job that is killed (or a disk that fills up mid-write) never leaves a shard
that looks complete but holds only part of its rows. Re-running the same corpus skips the shards that are
already recorded and only encodes the rest, and the resulting store is row-for-row identical to a run that
finished in one go.

Every store is bound to an :class:`EmbeddingSpec` whose fingerprint pins down what produced the vectors: the
model (name, revision and a digest of its architecture and weights), the resolved prompt/template, whether
embeddings are normalized, the dimension truncation, the output precision and the text normalization rule.
A store refuses to resume, extend or verify against a different fingerprint, and reports exactly which
field differs instead of silently mixing vectors.

A store can also be updated incrementally with a delta of upserts and deletes. Only the rows that actually
changed are re-encoded; untouched rows are copied from the existing shard files byte-for-byte, and unchanged
shards are reused as they are.

Example:
    >>> from sentence_transformers import SentenceTransformer
    >>> from sentence_transformers.util import EmbeddingStore
    >>> model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # doctest: +SKIP
    >>> store = EmbeddingStore("products-embeddings", shard_size=50_000)  # doctest: +SKIP
    >>> store.run(model, corpus, normalize_embeddings=True)  # doctest: +SKIP
    >>> ids, vectors = store.load()  # doctest: +SKIP
    >>> report = store.verify(model=model)  # doctest: +SKIP
    >>> report.ok  # doctest: +SKIP
    True
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
import unicodedata
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_SHARD_SIZE = 65_536
MANIFEST_FILE = "manifest.json"
LOCK_FILE = ".lock"
SHARDS_DIR = "shards"
_SPEC_FIELDS = (
    "model",
    "model_revision",
    "model_digest",
    "template",
    "task",
    "normalize_embeddings",
    "truncate_dim",
    "precision",
    "text_normalization",
)


class EmbeddingStoreError(Exception):
    """Base class for all embedding store errors."""


class FingerprintMismatchError(EmbeddingStoreError):
    """The vectors in the store were produced by a different encoder configuration.

    Attributes:
        differences: Human-readable descriptions of every differing fingerprint field.
    """

    def __init__(self, message: str, differences: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.differences = list(differences)


class PlanMismatchError(EmbeddingStoreError):
    """The store was built for a different corpus (different identifiers, texts or order).

    Attributes:
        differences: Human-readable descriptions of what differs, e.g. missing or extra identifiers.
    """

    def __init__(self, message: str, differences: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.differences = list(differences)


class StoreLockedError(EmbeddingStoreError):
    """Another process is writing this store and did not release it in time."""


class ShardClaimedError(EmbeddingStoreError):
    """A shard needed by this run is currently being written by another process."""


class StoreCorruptionError(EmbeddingStoreError):
    """A shard file exists but does not match its recorded size, digest or sidecar metadata."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8", errors="surrogatepass"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Flush a directory's entries so a rename survives a power loss (no-op where unsupported)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, never exposing a partial file under the final name."""
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _npy_array_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _model_identity(model: BaseModel) -> tuple[str, str | None]:
    """Best-effort ``(name, revision)`` for a model, falling back to its in-memory description."""
    card = getattr(model, "model_card_data", None)
    name = getattr(card, "base_model", None) if card is not None else None
    revision = getattr(card, "base_model_revision", None) if card is not None else None
    if not name:
        name = getattr(model, "name_or_path", None)
    if not name:
        for module in model.modules():
            config = getattr(module, "config", None)
            module_name = getattr(config, "name_or_path", None) if config is not None else None
            if module_name:
                name = module_name
                break
    if not name:
        name = f"<in-memory:{type(model).__name__}>"
    return str(name), str(revision) if revision else None


def _model_digest(model: BaseModel) -> str:
    """Digest the architecture, module configuration and weights so a model version cannot drift silently.

    Only persistent tensors are hashed (via ``state_dict``): non-persistent buffers are runtime caches that can
    change between calls without the model changing, and including them would make a legitimate resume look
    like a fingerprint mismatch. Wrapper prefixes added by ``torch.compile`` are normalized away for the same
    reason.
    """
    digest = hashlib.sha256()
    digest.update(_canonical_json({"type": type(model).__name__, "schema": SCHEMA_VERSION}).encode())
    for name, module in model.named_children():
        module_type = f"{type(module).__module__}.{type(module).__qualname__}"
        try:
            config = module.get_config_dict()
        except Exception:
            config = {}
        digest.update(_canonical_json({"name": name, "type": module_type, "config": config}).encode())
    try:
        state = model.state_dict()
    except Exception as exc:  # pragma: no cover - exotic backends with non-materialized weights
        logger.warning(
            f"Could not read the weights of {type(model).__name__} for its fingerprint ({exc}); the fingerprint "
            "will only cover the architecture and module configuration."
        )
        state = {}
    for key in sorted(state):
        tensor = state[key].detach()
        normalized_key = key.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
        digest.update(f"{normalized_key}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode())
        try:
            if tensor.dtype == torch.bfloat16:
                raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            else:
                raw = tensor.cpu().contiguous().numpy().tobytes()
        except Exception:  # pragma: no cover - meta/offloaded tensors
            digest.update(_canonical_json({"unreadable": str(tensor.dtype)}).encode())
            continue
        digest.update(raw)
    return digest.hexdigest()


def apply_text_normalization(text: str, rule: str) -> str:
    """Apply the named text normalization rule to a single input."""
    if rule == "none":
        return text
    if rule in ("nfc", "nfkc"):
        return unicodedata.normalize(rule.upper(), text)
    if rule == "nfc-strip-control":
        normalized = unicodedata.normalize("NFC", text)
        return "".join(
            char for char in normalized if char in ("\n", "\t") or unicodedata.category(char) not in ("Cc", "Cf")
        )
    raise ValueError(
        f"Unknown text_normalization rule {rule!r}. Choose one of: 'none', 'nfc', 'nfkc', 'nfc-strip-control'."
    )


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything besides the corpus that determines the produced vectors.

    The fingerprint of a spec is a single SHA-256 string over these fields. Two stores may only be resumed,
    extended or compared with one another when their fingerprints match.

    Attributes:
        model: Model name or in-memory description.
        model_revision: Resolved Hub revision, when available.
        model_digest: Digest over the model architecture, module configuration and weights.
        template: The resolved prompt/template prepended to every input, or None.
        task: The ``task`` passed to :meth:`~sentence_transformers.base.model.BaseModel.encode`.
        normalize_embeddings: Whether vectors were normalized to unit length.
        truncate_dim: Dimension truncation applied to the vectors.
        precision: Output precision passed to ``encode``.
        text_normalization: Name of the text normalization rule applied before encoding.
    """

    model: str
    model_revision: str | None = None
    model_digest: str = ""
    template: str | None = None
    task: str | None = None
    normalize_embeddings: bool = False
    truncate_dim: int | None = None
    precision: str = "float32"
    text_normalization: str = "none"

    def __post_init__(self) -> None:
        # Fail early on an unknown rule rather than on the first text.
        apply_text_normalization("", self.text_normalization)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _SPEC_FIELDS}

    @property
    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()).encode("utf-8"))

    def differences(self, other: EmbeddingSpec) -> list[str]:
        """List every fingerprint field that differs, in a form suitable for an error message."""
        mine, theirs = self.to_dict(), other.to_dict()
        return [
            f"{name}: store={mine[name]!r}, requested={theirs[name]!r}"
            for name in _SPEC_FIELDS
            if mine[name] != theirs[name]
        ]

    @classmethod
    def from_model(
        cls,
        model: BaseModel,
        *,
        prompt: str | None = None,
        prompt_name: str | None = None,
        task: str | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        precision: str = "float32",
        text_normalization: str = "none",
    ) -> EmbeddingSpec:
        """Build a spec for a model and the encode arguments that will be used with it."""
        resolve_prompt = getattr(model, "_resolve_prompt", None)
        resolved_template = resolve_prompt(prompt, prompt_name) if callable(resolve_prompt) else prompt
        if truncate_dim is None:
            truncate_dim = getattr(model, "truncate_dim", None)
        name, revision = _model_identity(model)
        return cls(
            model=name,
            model_revision=revision,
            model_digest=_model_digest(model),
            template=resolved_template,
            task=task,
            normalize_embeddings=bool(normalize_embeddings),
            truncate_dim=truncate_dim,
            precision=precision,
            text_normalization=text_normalization,
        )


@dataclass(frozen=True)
class CorpusPlan:
    """The deduplicated, deterministically ordered rows of one corpus.

    Row order is the order of first appearance in the input. Identifiers that appear more than once are
    kept once, at their first position, so a store never holds the same identifier twice.
    """

    ids: tuple[str, ...]
    text_hashes: tuple[str, ...]
    num_duplicates_dropped: int = 0

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def ids_hash(self) -> str:
        return _hash_lines(self.ids)

    @property
    def plan_hash(self) -> str:
        """Hash over identifiers, text hashes and their order."""
        return _hash_lines(f"{row_id}\x00{text_hash}" for row_id, text_hash in zip(self.ids, self.text_hashes))

    def differences(self, other: CorpusPlan, limit: int = 10) -> list[str]:
        differences: list[str] = []
        if len(self) != len(other):
            differences.append(f"num_rows: store={len(self)}, requested={len(other)}")
        mine, theirs = set(self.ids), set(other.ids)
        missing = [row_id for row_id in self.ids if row_id not in theirs][:limit]
        extra = [row_id for row_id in other.ids if row_id not in mine][:limit]
        if missing:
            differences.append(f"identifiers only in store (first {len(missing)}): {missing}")
        if extra:
            differences.append(f"identifiers only in the requested corpus (first {len(extra)}): {extra}")
        changed = [
            row_id
            for row_id, mine_hash, their_hash in zip(self.ids, self.text_hashes, other.text_hashes)
            if mine_hash != their_hash
        ][:limit]
        if changed:
            differences.append(f"texts changed for (first {len(changed)}): {changed}")
        if not differences and self.plan_hash != other.plan_hash:
            differences.append("row order differs")
        return differences


def _normalize_identifier(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _iter_corpus(corpus: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(identifier, text)`` pairs for every supported corpus form.

    Supported forms:
        * an in-memory sequence of texts, identified by ``"0"``, ``"1"``, ...;
        * a mapping or a sequence of ``(identifier, text)`` pairs;
        * a local text file (as a path, or a file-like object), read line by line with
          undecodable bytes replaced, identified by ``"line:0"``, ``"line:1"``, ...
    """
    if isinstance(corpus, (str, os.PathLike)):
        corpus = Path(corpus)
        with open(corpus, encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle):
                yield f"line:{line_number}", line.rstrip("\r\n")

    elif hasattr(corpus, "readline") and not isinstance(corpus, (list, tuple)):
        # A text file-like object, read lazily so huge files never need to fit in memory at once.
        for line_number, line in enumerate(corpus):
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            yield f"line:{line_number}", line.rstrip("\r\n")

    elif isinstance(corpus, Mapping):
        for key, value in corpus.items():
            yield _normalize_identifier(key), value

    else:
        for index, item in enumerate(corpus):
            if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], str):
                yield _normalize_identifier(item[0]), item[1]
            elif isinstance(item, str):
                yield str(index), item
            else:
                raise TypeError(
                    "Unsupported corpus item at position "
                    f"{index}: {type(item).__name__}. Pass texts, (identifier, text) pairs, a mapping "
                    "or a text file."
                )


def build_plan(corpus: Any, text_normalization: str = "none") -> CorpusPlan:
    """Build the deterministic row plan for a corpus without encoding anything."""
    ids: list[str] = []
    text_hashes: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for row_id, text in _iter_corpus(corpus):
        if row_id in seen:
            duplicates += 1
            continue
        seen.add(row_id)
        normalized = apply_text_normalization(text, text_normalization)
        ids.append(row_id)
        text_hashes.append(_sha256_text(normalized))
    return CorpusPlan(tuple(ids), tuple(text_hashes), duplicates)


@dataclass(frozen=True)
class ShardVerification:
    """The result of checking one shard file against its manifest entry.

    Attributes:
        shard: Shard index.
        num_rows: Rows the manifest records for the shard (``None`` when the entry is missing).
        first_row: First row index the shard should cover (``None`` when the entry is missing).
        problems: Human-readable descriptions of every problem found.
    """

    shard: int
    num_rows: int | None
    first_row: int | None
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class VerificationReport:
    """Everything :meth:`EmbeddingStore.verify` learned about a store without re-encoding anything.

    Attributes:
        path: Store directory.
        fingerprint: Fingerprint recorded in the manifest.
        expected_fingerprint: Fingerprint of the model/spec passed to ``verify``, if any.
        num_rows: Row count recorded in the manifest.
        num_shards: Number of shard entries recorded in the manifest.
        dim: Embedding dimension recorded in the manifest.
        shards: Per-shard verification results.
        differences: Every store-level problem found, in a form suitable for printing.
    """

    path: str
    fingerprint: str | None
    expected_fingerprint: str | None
    num_rows: int
    num_shards: int
    dim: int | None
    shards: list[ShardVerification] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.differences and all(shard.ok for shard in self.shards)

    def __str__(self) -> str:
        status = "OK" if self.ok else "INCOMPLETE"
        lines = [
            f"EmbeddingStore at {self.path}: {status}",
            f"  rows: {self.num_rows}",
            f"  shards: {self.num_shards}",
            f"  dim: {self.dim}",
            f"  fingerprint: {self.fingerprint or '<missing>'}",
        ]
        if self.expected_fingerprint and self.expected_fingerprint != self.fingerprint:
            lines.append(f"  expected fingerprint: {self.expected_fingerprint}")
        for difference in self.differences:
            lines.append(f"  - {difference}")
        for shard in self.shards:
            if not shard.ok:
                lines.append(f"  - shard {shard.shard}: {'; '.join(shard.problems)}")
        return "\n".join(lines)


@dataclass
class RunStats:
    """What one :meth:`EmbeddingStore.run` or :meth:`EmbeddingStore.apply_delta` call did.

    Attributes:
        num_rows: Rows in the store after the call.
        num_shards_total: Shards the row plan requires.
        num_shards_written: Shards written by this call.
        num_shards_reused: Shards that were already complete or left untouched.
        num_encoded: Rows encoded by this call.
        num_upserted: Rows passed to ``upserts`` (``apply_delta`` only).
        num_added: New rows (``apply_delta`` only).
        num_modified: Existing rows whose text changed and were re-encoded (``apply_delta`` only).
        num_deleted: Rows removed (``apply_delta`` only).
        num_unchanged: Upserted rows whose text was identical to the stored text (``apply_delta`` only).
        num_duplicates_dropped: Repeated identifiers discarded during the corpus scan.
    """

    num_rows: int = 0
    num_shards_total: int = 0
    num_shards_written: int = 0
    num_shards_reused: int = 0
    num_encoded: int = 0
    num_upserted: int = 0
    num_added: int = 0
    num_modified: int = 0
    num_deleted: int = 0
    num_unchanged: int = 0
    num_duplicates_dropped: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class _SpilledReader:
    """Read shard-sized, in-order slices of a corpus that was spilled to a JSON-lines file."""

    def __init__(self, path: Path, offsets: list[int]) -> None:
        self.path = path
        self.offsets = offsets

    @property
    def num_rows(self) -> int:
        return len(self.offsets) - 1

    def _read_between(self, start: int, end: int) -> list[bytes]:
        begin, stop = self.offsets[start], self.offsets[end]
        if stop <= begin:
            return []
        with open(self.path, "rb") as handle:
            handle.seek(begin)
            payload = handle.read(stop - begin)
        return [line for line in payload.split(b"\n") if line]

    def texts(self, start: int, end: int) -> list[str]:
        """Read rows ``[start, end)`` without scanning the rows before ``start``."""
        if start >= end:
            return []
        lines = self._read_between(start, end)
        if len(lines) != end - start:
            raise StoreCorruptionError(
                f"The corpus spill file {self.path} returned {len(lines)} rows for rows [{start}, {end})."
            )
        return [json.loads(line)[1] for line in lines]


class EmbeddingStore:
    """A directory of immutable embedding shards plus a manifest that describes them.

    The store is the unit of resumability. Each shard is written to a unique file, flushed and atomically
    renamed, and only then recorded in ``manifest.json`` (itself replaced atomically). A process killed at
    any point therefore leaves either the previous manifest or a manifest that references only complete
    shards — never a half-written shard that counts as done.

    Shard ownership is fixed by row range: shard ``i`` always covers rows
    ``[i * shard_size, (i + 1) * shard_size)`` of the deduplicated input order. Scheduling order therefore
    cannot move a row into a different shard, or into two shards, or into none.

    Args:
        path: Directory holding the store. It is created if it does not exist.
        shard_size: Number of rows per shard for a new store. Ignored when a manifest already exists.
        spec: Encoder specification for a new store. When omitted, the first :meth:`run` derives it from the
            model and its encode arguments.

    Raises:
        FingerprintMismatchError: ``spec`` was given but differs from an existing store's fingerprint.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        shard_size: int | None = None,
        spec: EmbeddingSpec | None = None,
    ) -> None:
        if shard_size is not None and shard_size <= 0:
            raise ValueError(f"shard_size must be a positive integer, got {shard_size}.")
        self.path = Path(path)
        self.shards_path = self.path / SHARDS_DIR
        self.manifest_path = self.path / MANIFEST_FILE
        self._lock_path = self.path / LOCK_FILE
        self._manifest: dict[str, Any] | None = None
        self._new_shard_size = shard_size or DEFAULT_SHARD_SIZE
        if self.manifest_path.exists():
            self._manifest = self._read_manifest()
            if spec is not None and spec.fingerprint != self._manifest["fingerprint"]:
                differences = self.spec.differences(spec)
                raise FingerprintMismatchError(
                    f"EmbeddingStore at {self.path} was created with a different encoder fingerprint "
                    f"({self._manifest['fingerprint']}). Refusing to reuse its vectors.\n"
                    + "\n".join(f"  - {difference}" for difference in differences),
                    differences,
                )
            if shard_size is not None and shard_size != self._manifest["shard_size"]:
                logger.warning(
                    f"Ignoring shard_size={shard_size}: the existing store uses {self._manifest['shard_size']}."
                )
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.shards_path.mkdir(parents=True, exist_ok=True)
            if spec is not None:
                self._manifest = self._new_manifest(spec, self._new_shard_size)
                self._create_manifest(self._manifest)

    def __repr__(self) -> str:
        rows = self._manifest["num_rows"] if self._manifest else None
        return f"EmbeddingStore({str(self.path)!r}, rows={rows})"

    # ------------------------------------------------------------------ manifest

    @property
    def manifest(self) -> dict[str, Any]:
        """The manifest as this instance last read or wrote it."""
        if self._manifest is None:
            if not self.manifest_path.exists():
                raise EmbeddingStoreError(f"No embedding store manifest at {self.manifest_path}.")
            self._manifest = self._read_manifest()
        return self._manifest

    @property
    def spec(self) -> EmbeddingSpec:
        raw = self.manifest["spec"]
        return EmbeddingSpec(**{name: raw[name] for name in _SPEC_FIELDS})

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]

    @property
    def num_rows(self) -> int:
        return self.manifest["num_rows"]

    @property
    def shard_size(self) -> int:
        return self.manifest["shard_size"] if self.manifest_path.exists() else self._new_shard_size

    @property
    def num_shards_expected(self) -> int:
        return math.ceil(self.num_rows / self.shard_size) if self.num_rows else 0

    def completed_shards(self) -> dict[int, dict[str, Any]]:
        """Manifest entries of shards that are fully written, keyed by shard index.

        An entry only counts once its data file exists and holds exactly the recorded number of rows. A shard
        whose file was truncated (for example by a disk that filled up) or lost is reported as pending again,
        so the next :meth:`run` re-encodes it instead of trusting the manifest blindly.
        """
        completed: dict[int, dict[str, Any]] = {}
        for entry in self.manifest["shards"]:
            if not self._entry_is_complete(entry):
                logger.warning(
                    f"Shard {entry['shard']} of {self.path} is recorded in the manifest but its data file is "
                    "missing, truncated or unreadable; it will be re-encoded."
                )
                continue
            completed[entry["shard"]] = entry
        return completed

    def _reload_completed_shards(self) -> dict[int, dict[str, Any]]:
        """Re-read the manifest and return the complete shards, for cross-process coordination."""
        self._reload()
        return self.completed_shards()

    def _entry_is_complete(self, entry: dict[str, Any]) -> bool:
        data_path = self.path / entry["data"]
        meta_path = self.path / entry["meta"]
        if not data_path.exists() or not meta_path.exists():
            return False
        try:
            array = np.load(data_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return False
        try:
            return array.ndim == 2 and int(array.shape[0]) == int(entry["num_rows"])
        finally:
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()

    def pending_shards(self) -> list[int]:
        """Indices of shards not yet fully written, in ascending order."""
        done = self.completed_shards()
        return [index for index in range(self.num_shards_expected) if index not in done]

    @staticmethod
    def _new_manifest(spec: EmbeddingSpec, shard_size: int) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": spec.fingerprint,
            "spec": spec.to_dict(),
            "shard_size": shard_size,
            "num_rows": 0,
            "num_shards": 0,
            "dim": None,
            "dtype": "float32",
            "plan_hash": None,
            "ids_hash": None,
            "num_duplicates_dropped": 0,
            "shards": [],
            "created_at": _now(),
            "updated_at": _now(),
        }

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = _read_json(self.manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreCorruptionError(f"Could not read {self.manifest_path}: {exc}") from exc
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise StoreCorruptionError(
                f"Unsupported embedding store schema {manifest.get('schema_version')!r} in {self.manifest_path}; "
                f"this version supports schema {SCHEMA_VERSION}."
            )
        return manifest

    def _reload(self) -> dict[str, Any]:
        self._manifest = self._read_manifest()
        return self._manifest

    def _require_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise EmbeddingStoreError(
                f"No embedding store at {self.path}. Call run() with a model first, or pass spec= to create it."
            )
        return self.manifest

    def _create_manifest(self, manifest: dict[str, Any]) -> None:
        """Create the manifest unless another process already did, atomically either way.

        The manifest is only ever visible complete: it is written to a temporary file and linked into place
        with ``os.link``, which fails (instead of overwriting) when a concurrent process won the race.
        """
        tmp_path = self.path / f".manifest.json.create-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            _atomic_write_bytes(tmp_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode())
            try:
                os.link(tmp_path, self.manifest_path)
            except FileExistsError:
                self._manifest = self._read_manifest()
                return
            self._manifest = manifest
        finally:
            tmp_path.unlink(missing_ok=True)
        _fsync_dir(self.path)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _now()
        _atomic_write_json(self.manifest_path, manifest)
        self._manifest = manifest

    # ------------------------------------------------------------------ locking

    @contextmanager
    def _advisory_lock(self, path: Path, *, blocking: bool, timeout: float | None = 60.0) -> Iterator[bool]:
        """Hold an advisory exclusive lock on ``path`` for the duration of the context.

        The kernel releases these locks when a process dies, so a killed job never leaves the store or a shard
        permanently claimed. Yields whether the lock was acquired.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+b")
        acquired = False
        try:
            deadline = time.monotonic() + (timeout if timeout is not None else 60.0)
            while True:
                acquired = self._try_lock(handle)
                if acquired or not blocking or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                self._unlock(handle)
            handle.close()

    @staticmethod
    def _try_lock(handle: Any) -> bool:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    @staticmethod
    def _unlock(handle: Any) -> None:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _store_lock(self, *, blocking: bool = True, timeout: float | None = 60.0):
        return self._advisory_lock(self._lock_path, blocking=blocking, timeout=timeout)

    def _claim_path(self, shard: int) -> Path:
        return self.shards_path / f".claim-{shard:05d}"

    # ------------------------------------------------------------------ file naming

    @staticmethod
    def _shard_file_names(shard: int, data_sha256: str, ids_sha256: str) -> tuple[str, str]:
        # The name covers both the vectors and the identifiers: two generations can hold identical vectors for
        # different rows (for example when a row moves between shards), and must not share a file.
        stem = f"shard-{shard:05d}-{data_sha256[:12]}-{ids_sha256[:12]}"
        return f"{stem}.npy", f"{stem}.json"

    # ------------------------------------------------------------------ run

    def run(
        self,
        model: BaseModel,
        corpus: Any,
        *,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        shard_ids: Sequence[int] | None = None,
        prompt: str | None = None,
        prompt_name: str | None = None,
        task: str | None = None,
        normalize_embeddings: bool | None = None,
        truncate_dim: int | None = None,
        precision: str | None = None,
        text_normalization: str | None = None,
        device: str | list[str] | None = None,
        pool: dict[str, Any] | None = None,
        lock_timeout: float = 60.0,
        **encode_kwargs: Any,
    ) -> RunStats:
        """Encode ``corpus`` into the store, skipping shards that are already complete.

        Repeated calls resume where a previous (possibly killed) call stopped. The row order, row count and
        identifier set of the result are identical to encoding the whole corpus in one go, because shard
        ownership is fixed by row range rather than by scheduling.

        Only one shard's vectors are held in the encoding loop at a time, so peak memory is bounded by the
        corpus texts plus one shard, not by the full output matrix.

        Args:
            model: Model to encode with.
            corpus: A sequence of texts, a sequence of ``(identifier, text)`` pairs, a mapping of identifier
                to text, or a path to a local UTF-8 text file with one text per line. Undecodable bytes in a
                file are replaced rather than raising.
            batch_size: Batch size passed to ``encode``.
            show_progress_bar: Whether to show a progress bar. Defaults to the logger's verbosity.
            shard_ids: Only encode these shard indices. Several workers or machines may encode disjoint
                shards of one store safely; each shard is claimed with a kernel lock while being written.
            prompt: Prompt/template to resolve and use. Defaults to the store's recorded template.
            prompt_name: Prompt name resolved through the model's prompts.
            task: The ``task`` argument for ``encode``. Defaults to the store's recorded task.
            normalize_embeddings: Whether to L2-normalize. Defaults to the store's recorded setting.
            truncate_dim: Dimension truncation. Defaults to the store's recorded setting.
            precision: Output precision. Defaults to the store's recorded setting.
            text_normalization: Text normalization rule (``"none"``, ``"nfc"``, ``"nfkc"`` or
                ``"nfc-strip-control"``). Defaults to the store's recorded setting.
            device: Device or device list passed to ``encode``.
            pool: A multi-process pool passed to ``encode``.
            lock_timeout: Seconds to wait for the store lock when recording a finished shard.
            **encode_kwargs: Extra arguments forwarded to ``encode`` (for example ``processing_kwargs``).

        Returns:
            RunStats describing what was written and what was reused.

        Raises:
            FingerprintMismatchError: The model or encode settings do not match the store's fingerprint.
            PlanMismatchError: The corpus rows, texts or order differ from the store's plan.
            ShardClaimedError: A shard that still needs encoding is being written by another process.
            StoreLockedError: The store lock could not be acquired in time.
        """
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        if not self.manifest_path.exists():
            spec = EmbeddingSpec.from_model(
                model,
                prompt=prompt,
                prompt_name=prompt_name,
                task=task,
                normalize_embeddings=bool(normalize_embeddings),
                truncate_dim=truncate_dim,
                precision=precision or "float32",
                text_normalization=text_normalization or "none",
            )
            self._create_manifest(self._new_manifest(spec, self._new_shard_size))
            if self.manifest["fingerprint"] != spec.fingerprint:
                self._assert_fingerprint(self.spec, spec)

        manifest = self._require_manifest()
        stored = self.spec
        candidate = self._candidate_spec(
            model,
            prompt=prompt,
            prompt_name=prompt_name,
            task=task,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            precision=precision,
            text_normalization=text_normalization,
        )
        self._assert_fingerprint(stored, candidate)

        with self._spilled_corpus(corpus, stored.text_normalization) as (reader, plan):
            return self._run_plan(
                model=model,
                reader=reader,
                plan=plan,
                manifest=manifest,
                stored=stored,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                shard_ids=shard_ids,
                device=device,
                pool=pool,
                lock_timeout=lock_timeout,
                encode_kwargs=encode_kwargs,
            )

    def _run_plan(
        self,
        *,
        model: BaseModel,
        reader: _SpilledReader,
        plan: CorpusPlan,
        manifest: dict[str, Any],
        stored: EmbeddingSpec,
        batch_size: int,
        show_progress_bar: bool | None,
        shard_ids: Sequence[int] | None,
        device: str | list[str] | None,
        pool: dict[str, Any] | None,
        lock_timeout: float,
        encode_kwargs: dict[str, Any],
    ) -> RunStats:
        """Resume or finish the store for an already-scanned plan, encoding shard by shard."""
        total_shards = math.ceil(len(plan) / manifest["shard_size"]) if len(plan) else 0
        self._adopt_or_check_plan(plan, total_shards)

        done = self.completed_shards()
        todo = [index for index in range(total_shards) if index not in done]
        if shard_ids is not None:
            requested = {int(index) for index in shard_ids}
            invalid = sorted(index for index in requested if index < 0 or index >= total_shards)
            if invalid:
                raise ValueError(
                    f"shard_ids {invalid} are outside the store's {total_shards} shards "
                    f"(rows={len(plan)}, shard_size={manifest['shard_size']})."
                )
            todo = [index for index in todo if index in requested]

        stats = RunStats(
            num_rows=len(plan),
            num_shards_total=total_shards,
            num_duplicates_dropped=plan.num_duplicates_dropped,
        )
        if not todo:
            stats.num_shards_reused = total_shards
            return stats

        self._reject_fingerprint_kwargs(encode_kwargs)
        encode_args = dict(
            prompt=stored.template,
            task=stored.task,
            normalize_embeddings=stored.normalize_embeddings,
            truncate_dim=stored.truncate_dim,
            precision=stored.precision,
            **encode_kwargs,
        )
        if device is not None:
            encode_args["device"] = device
        if pool is not None:
            encode_args["pool"] = pool

        iterator: Iterable[int] = todo
        if show_progress_bar:
            from tqdm.autonotebook import tqdm

            iterator = tqdm(todo, desc="Shards")

        shard_size = manifest["shard_size"]
        for shard in iterator:
            with self._advisory_lock(self._claim_path(shard), blocking=False) as claimed:
                if not claimed:
                    raise ShardClaimedError(
                        f"Shard {shard} of {self.path} is currently being written by another process."
                    )
                # Reload after claiming: a worker that finished while we waited for the claim recorded its entry.
                if shard in self._reload_completed_shards():
                    stats.num_shards_reused += 1
                    continue
                start = shard * shard_size
                end = min(start + shard_size, len(plan))
                vectors = model.encode(
                    reader.texts(start, end),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    **encode_args,
                )
                if len(vectors) != end - start:
                    raise EmbeddingStoreError(
                        f"encode() returned {len(vectors)} vectors for the {end - start} rows of shard {shard}."
                    )
                entry = self._write_shard(shard, start, plan, vectors)
                self._record_shard(shard, entry, lock_timeout=lock_timeout)
                stats.num_shards_written += 1
                stats.num_encoded += end - start

        stats.num_shards_reused = total_shards - stats.num_shards_written - len(self.pending_shards())
        return stats

    # ------------------------------------------------------------------ resume helpers

    def _candidate_spec(
        self,
        model: BaseModel,
        *,
        prompt: str | None,
        prompt_name: str | None,
        task: str | None,
        normalize_embeddings: bool | None,
        truncate_dim: int | None,
        precision: str | None,
        text_normalization: str | None,
    ) -> EmbeddingSpec:
        """The spec the caller asks for, with unspecified settings inherited from the store."""
        stored = self.spec
        resolve_prompt = getattr(model, "_resolve_prompt", None)
        if prompt is None and prompt_name is None:
            template = stored.template
        else:
            template = resolve_prompt(prompt, prompt_name) if callable(resolve_prompt) else prompt
        name, revision = _model_identity(model)
        return EmbeddingSpec(
            model=name,
            model_revision=revision if revision is not None else stored.model_revision,
            model_digest=_model_digest(model),
            template=template,
            task=task if task is not None else stored.task,
            normalize_embeddings=(
                bool(normalize_embeddings) if normalize_embeddings is not None else stored.normalize_embeddings
            ),
            truncate_dim=truncate_dim if truncate_dim is not None else stored.truncate_dim,
            precision=precision if precision is not None else stored.precision,
            text_normalization=text_normalization if text_normalization is not None else stored.text_normalization,
        )

    @staticmethod
    def _assert_fingerprint(stored: EmbeddingSpec, candidate: EmbeddingSpec) -> None:
        if stored.fingerprint == candidate.fingerprint:
            return
        differences = stored.differences(candidate)
        raise FingerprintMismatchError(
            "Refusing to reuse or extend this embedding store: the fingerprint of what would produce the "
            "vectors differs from the fingerprint the store was built with. Re-encode into a new store, or "
            "restore the original model, template and settings.\n"
            + "\n".join(f"  - {difference}" for difference in differences),
            differences,
        )

    @staticmethod
    def _reject_fingerprint_kwargs(encode_kwargs: dict[str, Any]) -> None:
        reserved = {
            "prompt",
            "prompt_name",
            "task",
            "normalize_embeddings",
            "truncate_dim",
            "precision",
            "convert_to_numpy",
            "convert_to_tensor",
            "batch_size",
            "show_progress_bar",
        }
        duplicated = sorted(reserved & set(encode_kwargs))
        if duplicated:
            raise TypeError(
                f"{duplicated} determine the store fingerprint and must be passed as explicit keyword arguments "
                "of run()/apply_delta(), not through **encode_kwargs."
            )

    @contextmanager
    def _spilled_corpus(self, corpus: Any, text_normalization: str) -> Iterator[tuple[Any, CorpusPlan]]:
        """Scan the corpus into a temporary JSON-lines file and yield a shard-wise reader plus its plan.

        This keeps peak memory bounded by the row identifiers and their hashes rather than by every text at
        once, which matters for corpora in the hundreds of thousands of rows. The spill file lives in the
        store directory (so it is on the same disk as the output) and is removed on exit, including after a
        crash in the caller.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        fd, spill_name = tempfile.mkstemp(prefix=".corpus-", suffix=".jsonl", dir=self.path)
        os.close(fd)
        spill_path = Path(spill_name)
        ids: list[str] = []
        text_hashes: list[str] = []
        seen: set[str] = set()
        duplicates = 0
        # Byte offsets let any shard be read directly instead of rescanning the whole corpus per shard.
        offsets: list[int] = [0]
        try:
            with open(spill_path, "wb") as handle:
                for row_id, text in _iter_corpus(corpus):
                    if row_id in seen:
                        duplicates += 1
                        continue
                    seen.add(row_id)
                    normalized = apply_text_normalization(text, text_normalization)
                    ids.append(row_id)
                    text_hashes.append(_sha256_text(normalized))
                    line = json.dumps([row_id, normalized], ensure_ascii=True).encode("utf-8") + b"\n"
                    handle.write(line)
                    offsets.append(offsets[-1] + len(line))
                handle.flush()
                os.fsync(handle.fileno())
            plan = CorpusPlan(tuple(ids), tuple(text_hashes), duplicates)
            yield _SpilledReader(spill_path, offsets), plan
        finally:
            spill_path.unlink(missing_ok=True)

    def _stored_plan(self) -> CorpusPlan:
        """The full row plan, including rows whose shard has not been written yet."""
        plan_file = self.manifest.get("plan_file")
        if plan_file:
            payload = _read_json(self.path / plan_file)
            return CorpusPlan(tuple(payload["ids"]), tuple(payload["text_sha256"]))
        ids: list[str] = []
        hashes: list[str] = []
        for meta in self._stored_metas():
            ids.extend(meta["ids"])
            hashes.extend(meta["text_sha256"])
        return CorpusPlan(tuple(ids), tuple(hashes))

    def _write_plan_file(self, plan: CorpusPlan) -> str:
        """Persist the full plan so a partially written store can still recognise its corpus."""
        name = f"plan-{plan.plan_hash[:16]}.json"
        payload = {
            "ids_hash": plan.ids_hash,
            "plan_hash": plan.plan_hash,
            "ids": list(plan.ids),
            "text_sha256": list(plan.text_hashes),
        }
        _atomic_write_json(self.shards_path / name, payload)
        return f"{SHARDS_DIR}/{name}"

    def _adopt_or_check_plan(self, plan: CorpusPlan, total_shards: int) -> None:
        """Adopt this plan for a fresh store, or require the stored plan to match exactly."""
        manifest = self._reload()
        if manifest["num_rows"] == 0 and not manifest["shards"]:
            self._adopt_plan(plan, total_shards)
            return
        stored = self._stored_plan()
        if stored.plan_hash == plan.plan_hash:
            if manifest["plan_hash"] != plan.plan_hash or "plan_file" not in manifest:
                self._adopt_plan(plan, total_shards)
            return
        if stored.ids_hash == plan.ids_hash:
            changed = [
                row_id for row_id, old, new in zip(stored.ids, stored.text_hashes, plan.text_hashes) if old != new
            ]
            differences = [f"texts changed for (first {min(len(changed), 10)}): {changed[:10]}"]
            raise PlanMismatchError(
                "The corpus changed the text of rows that are already stored. Pass those rows to apply_delta() "
                "to re-encode only them instead of resuming with a different corpus.\n"
                + "\n".join(f"  - {difference}" for difference in differences),
                differences,
            )
        differences = stored.differences(plan)
        raise PlanMismatchError(
            "Refusing to resume this store for a different corpus: the row count, identifiers or their order "
            "must match the previous run exactly. Use apply_delta() to add or remove rows.\n"
            + "\n".join(f"  - {difference}" for difference in differences),
            differences,
        )

    def _adopt_plan(self, plan: CorpusPlan, total_shards: int) -> None:
        """Persist a newly adopted plan (and its sidecar file) under the store lock."""
        plan_file = self._write_plan_file(plan)
        with self._store_lock() as acquired:
            if not acquired:
                raise StoreLockedError(f"Could not acquire the lock for {self.path} to record the corpus plan.")
            manifest = self._reload()
            if manifest["num_rows"] != 0 and manifest["plan_hash"] != plan.plan_hash:
                # Another process adopted a different plan while we were waiting for the lock.
                raise PlanMismatchError(
                    f"The store at {self.path} adopted a different corpus plan concurrently; re-run to resume it."
                )
            manifest.update(
                num_rows=len(plan),
                num_shards=total_shards,
                plan_hash=plan.plan_hash,
                ids_hash=plan.ids_hash,
                num_duplicates_dropped=plan.num_duplicates_dropped,
                plan_file=plan_file,
            )
            self._write_manifest(manifest)

    # ------------------------------------------------------------------ shard writing

    @staticmethod
    def _encode_vectors(vectors: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(vectors)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def _write_shard(self, shard: int, first_row: int, plan: CorpusPlan, vectors: np.ndarray) -> dict[str, Any]:
        """Atomically write one shard's data and sidecar metadata, returning its manifest entry."""
        array = self._encode_vectors(vectors)
        data = _npy_array_bytes(array)
        data_sha256 = _sha256_bytes(data)
        rows = int(array.shape[0])
        ids = list(plan.ids[first_row : first_row + rows])
        ids_sha256 = _hash_lines(ids)
        data_name, meta_name = self._shard_file_names(shard, data_sha256, ids_sha256)
        meta = {
            "shard": shard,
            "first_row": first_row,
            "num_rows": rows,
            "dim": int(array.shape[1]),
            "dtype": str(array.dtype),
            "ids": ids,
            "text_sha256": list(plan.text_hashes[first_row : first_row + rows]),
            "ids_sha256": ids_sha256,
            "data_sha256": data_sha256,
        }
        _atomic_write_bytes(self.shards_path / data_name, data)
        _atomic_write_json(self.shards_path / meta_name, meta)
        return {
            "shard": shard,
            "first_row": first_row,
            "num_rows": rows,
            "dim": int(array.shape[1]),
            "dtype": str(array.dtype),
            "data": f"{SHARDS_DIR}/{data_name}",
            "meta": f"{SHARDS_DIR}/{meta_name}",
            "data_sha256": data_sha256,
            "ids_sha256": meta["ids_sha256"],
            "plan_hash": plan.plan_hash,
        }

    def _record_shard(self, shard: int, entry: dict[str, Any], *, lock_timeout: float = 60.0) -> None:
        """Record a fully written shard in the manifest, replacing any previous entry for that index.

        The update is skipped if the manifest's plan changed after this shard was encoded (for example because
        a delta replaced the layout), so a stale shard can never be grafted onto a newer store.
        """
        with self._store_lock(timeout=lock_timeout) as acquired:
            if not acquired:
                raise StoreLockedError(f"Could not acquire the lock for {self.path} while recording shard {shard}.")
            manifest = self._reload()
            if manifest["plan_hash"] is not None and manifest["plan_hash"] != entry["plan_hash"]:
                raise PlanMismatchError(
                    f"The corpus plan of {self.path} changed while shard {shard} was being encoded, so its "
                    "vectors no longer belong to the store's current layout. Re-run to encode the current plan."
                )
            entries = [existing for existing in manifest["shards"] if existing["shard"] != shard]
            entries.append({key: value for key, value in entry.items() if key != "plan_hash"})
            entries.sort(key=lambda item: item["shard"])
            manifest["shards"] = entries
            manifest["dim"] = entry["dim"]
            manifest["dtype"] = entry["dtype"]
            self._write_manifest(manifest)

    # ------------------------------------------------------------------ reading

    def _sorted_entries(self) -> list[dict[str, Any]]:
        return sorted(self.manifest["shards"], key=lambda item: item["shard"])

    def _stored_metas(self) -> list[dict[str, Any]]:
        return [_read_json(self.path / entry["meta"]) for entry in self._sorted_entries()]

    def get_ids(self) -> list[str]:
        """Return the stored identifiers in row order, without loading vectors."""
        self._require_manifest()
        ids: list[str] = []
        for meta in self._stored_metas():
            ids.extend(meta["ids"])
        return ids

    def load(self, *, verify_checksums: bool = False) -> tuple[list[str], np.ndarray]:
        """Return ``(identifiers, vectors)`` for the whole store, in row order.

        Args:
            verify_checksums: Recompute each shard's SHA-256 as well. Slower, but detects silent corruption.

        Raises:
            EmbeddingStoreError: The store is incomplete (some shards are missing).
            StoreCorruptionError: A shard file does not match its recorded row count or digest.
        """
        self._require_manifest()
        pending = self.pending_shards()
        if pending:
            raise EmbeddingStoreError(
                f"The store at {self.path} is incomplete: {len(pending)} of {self.num_shards_expected} shards "
                f"are missing (first: {pending[:5]}). Resume it with run() first."
            )
        ids: list[str] = []
        blocks: list[np.ndarray] = []
        for entry in self._sorted_entries():
            meta = _read_json(self.path / entry["meta"])
            data_path = self.path / entry["data"]
            array = np.load(data_path, allow_pickle=False)
            if array.shape[0] != entry["num_rows"]:
                raise StoreCorruptionError(
                    f"{entry['data']} holds {array.shape[0]} rows but the manifest records {entry['num_rows']}."
                )
            if len(meta["ids"]) != array.shape[0]:
                raise StoreCorruptionError(
                    f"{entry['meta']} lists {len(meta['ids'])} identifiers for {array.shape[0]} rows."
                )
            if verify_checksums and _sha256_bytes(data_path.read_bytes()) != entry["data_sha256"]:
                raise StoreCorruptionError(f"{entry['data']} does not match its recorded SHA-256.")
            ids.extend(meta["ids"])
            blocks.append(array)
        if not blocks:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.concatenate(blocks, axis=0)

    # ------------------------------------------------------------------ incremental updates

    def apply_delta(
        self,
        model: BaseModel,
        *,
        upserts: Any = None,
        deletes: Iterable[str] = (),
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        prompt: str | None = None,
        prompt_name: str | None = None,
        task: str | None = None,
        normalize_embeddings: bool | None = None,
        truncate_dim: int | None = None,
        precision: str | None = None,
        text_normalization: str | None = None,
        device: str | list[str] | None = None,
        pool: dict[str, Any] | None = None,
        lock_timeout: float = 60.0,
        **encode_kwargs: Any,
    ) -> RunStats:
        """Apply additions, modifications and deletions, re-encoding only the affected rows.

        The result is laid out as if the whole corpus (the stored rows in their original order, followed by
        the added rows in input order) had been encoded in one go: modified rows keep their position, deleted
        rows disappear, and untouched rows are copied from the existing shards instead of being re-encoded.

        Old vectors are read through memory maps and new shards are written one at a time, so the update never
        holds the whole vector matrix in memory. Shards with no change at all are reused as they are.

        The update becomes visible in one atomic manifest replacement. A crash before it leaves the previous
        store fully intact; a crash after it leaves the new store plus unreferenced files that the next
        :meth:`prune` (called at the end of every delta) removes.

        Args:
            model: The model to encode with. Must match the store's fingerprint.
            upserts: New or modified rows, in any corpus form accepted by :meth:`run`. Rows whose normalized
                text is unchanged are left untouched.
            deletes: Identifiers to remove from the store.
            batch_size: Batch size passed to ``encode``.
            show_progress_bar: Whether to show a progress bar.
            prompt: Prompt/template. Defaults to the store's recorded template.
            prompt_name: Prompt name resolved through the model's prompts.
            task: The ``task`` argument for ``encode``. Defaults to the store's recorded task.
            normalize_embeddings: Whether to L2-normalize. Defaults to the store's recorded setting.
            truncate_dim: Dimension truncation. Defaults to the store's recorded setting.
            precision: Output precision. Defaults to the store's recorded setting.
            text_normalization: Text normalization rule. Defaults to the store's recorded setting.
            device: Device or device list passed to ``encode``.
            pool: A multi-process pool passed to ``encode``.
            lock_timeout: Seconds to wait for the store lock.
            **encode_kwargs: Extra arguments forwarded to ``encode``.

        Returns:
            RunStats describing what changed.

        Raises:
            FingerprintMismatchError: The model does not match the store's fingerprint.
            ValueError: A deleted identifier is not stored, an identifier is both deleted and upserted, or an
                identifier appears more than once in ``upserts``.
            EmbeddingStoreError: The store has no rows yet; call :meth:`run` first.
            PlanMismatchError: Another process replaced the store's layout while this delta was prepared.
        """
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        self._require_manifest()
        if self.manifest["num_rows"] == 0 and not self.manifest["shards"]:
            raise EmbeddingStoreError(
                f"The store at {self.path} is still empty. Call run() to build it before applying a delta."
            )
        pending = self.pending_shards()
        if pending:
            raise EmbeddingStoreError(
                f"The store at {self.path} is still incomplete: {len(pending)} of {self.num_shards_expected} "
                f"shards are missing (first: {pending[:5]}). Resume it with run() before applying a delta."
            )
        self._assert_fingerprint(
            self.spec,
            self._candidate_spec(
                model,
                prompt=prompt,
                prompt_name=prompt_name,
                task=task,
                normalize_embeddings=normalize_embeddings,
                truncate_dim=truncate_dim,
                precision=precision,
                text_normalization=text_normalization,
            ),
        )

        old_plan = self._stored_plan()
        if len(old_plan) != self.manifest["num_rows"]:
            raise StoreCorruptionError(
                f"The manifest records {self.manifest['num_rows']} rows but the row plan holds {len(old_plan)}."
            )
        old_hash_by_id = dict(zip(old_plan.ids, old_plan.text_hashes))
        delete_set = {_normalize_identifier(row_id) for row_id in deletes}
        unknown = sorted(delete_set - set(old_plan.ids))
        if unknown:
            raise ValueError(
                f"Cannot delete identifiers that are not stored: {unknown[:10]}"
                + (" ..." if len(unknown) > 10 else "")
            )

        rule = self.spec.text_normalization
        upsert_ids: list[str] = []
        upsert_texts: dict[str, str] = {}
        upsert_hashes: dict[str, str] = {}
        for row_id, text in _iter_corpus(upserts) if upserts is not None else ():
            if row_id in delete_set:
                raise ValueError(f"Identifier {row_id!r} is both deleted and upserted in the same delta.")
            if row_id in upsert_texts:
                raise ValueError(
                    f"Identifier {row_id!r} appears more than once in the upsert batch; a delta must be unambiguous."
                )
            normalized = apply_text_normalization(text, rule)
            upsert_ids.append(row_id)
            upsert_texts[row_id] = normalized
            upsert_hashes[row_id] = _sha256_text(normalized)

        added = [row_id for row_id in upsert_ids if row_id not in old_hash_by_id]
        modified = [
            row_id
            for row_id in upsert_ids
            if row_id in old_hash_by_id and old_hash_by_id[row_id] != upsert_hashes[row_id]
        ]
        unchanged = [
            row_id
            for row_id in upsert_ids
            if row_id in old_hash_by_id and old_hash_by_id[row_id] == upsert_hashes[row_id]
        ]

        new_ids = [row_id for row_id in old_plan.ids if row_id not in delete_set] + added
        new_hashes = [
            upsert_hashes[row_id] if row_id in upsert_hashes else old_hash_by_id[row_id] for row_id in new_ids
        ]
        if len(new_ids) != len(set(new_ids)):  # pragma: no cover - guarded by the checks above
            raise EmbeddingStoreError(f"The resulting row plan for {self.path} contains duplicate identifiers.")

        stats = RunStats(
            num_rows=len(new_ids),
            num_upserted=len(upsert_ids),
            num_added=len(added),
            num_modified=len(modified),
            num_deleted=len(delete_set),
            num_unchanged=len(unchanged),
        )

        reencode = [row_id for row_id in upsert_ids if row_id in set(added) | set(modified)]
        changed_vectors: dict[str, np.ndarray] = {}
        self._reject_fingerprint_kwargs(encode_kwargs)
        if reencode:
            stored = self.spec
            encode_args = dict(
                prompt=stored.template,
                task=stored.task,
                normalize_embeddings=stored.normalize_embeddings,
                truncate_dim=stored.truncate_dim,
                precision=stored.precision,
                **encode_kwargs,
            )
            if device is not None:
                encode_args["device"] = device
            if pool is not None:
                encode_args["pool"] = pool
            vectors = model.encode(
                [upsert_texts[row_id] for row_id in reencode],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
                **encode_args,
            )
            if len(vectors) != len(reencode):
                raise EmbeddingStoreError(
                    f"encode() returned {len(vectors)} vectors for {len(reencode)} added or modified rows."
                )
            changed_vectors = {row_id: vectors[index] for index, row_id in enumerate(reencode)}

        plan = CorpusPlan(tuple(new_ids), tuple(new_hashes))
        self._rewrite_store(
            plan=plan,
            old_plan=old_plan,
            changed_vectors=changed_vectors,
            modified=set(modified),
            expected_plan_hash=old_plan.plan_hash,
        )
        self.prune()
        return stats

    def _rewrite_store(
        self,
        *,
        plan: CorpusPlan,
        old_plan: CorpusPlan,
        changed_vectors: dict[str, np.ndarray],
        modified: set[str],
        expected_plan_hash: str,
    ) -> None:
        """Write a new shard generation for ``plan``, then swap the manifest atomically.

        ``expected_plan_hash`` is the plan the delta was computed against. If another writer replaced the
        store's layout in the meantime, the swap is refused rather than silently dropping its changes or
        grafting this delta onto a layout it was not computed for.
        """
        # Hold the store lock for the whole rewrite so another delta cannot rewrite the layout concurrently.
        # Rewriting shard files does not hold the GIL, so this is comfortable even for hundreds of shards.
        with self._store_lock() as acquired:
            if not acquired:
                raise StoreLockedError(f"Could not acquire the lock for {self.path} to apply the delta.")
            self._reload()
            self._rewrite_store_locked(
                plan=plan,
                old_plan=old_plan,
                changed_vectors=changed_vectors,
                modified=modified,
                expected_plan_hash=expected_plan_hash,
            )

    def _rewrite_store_locked(
        self,
        *,
        plan: CorpusPlan,
        old_plan: CorpusPlan,
        changed_vectors: dict[str, np.ndarray],
        modified: set[str],
        expected_plan_hash: str,
    ) -> None:
        """The body of :meth:`_rewrite_store`, called with the store lock held."""
        shard_size = self.manifest["shard_size"]
        old_entries = self._sorted_entries()
        old_meta_by_shard = {entry["shard"]: _read_json(self.path / entry["meta"]) for entry in old_entries}
        old_rows_by_shard = {shard: tuple(meta["ids"]) for shard, meta in old_meta_by_shard.items()}
        old_index_by_id = {row_id: index for index, row_id in enumerate(old_plan.ids)}
        old_entry_of_row: dict[int, tuple[dict[str, Any], int]] = {}
        for entry in old_entries:
            for offset in range(entry["num_rows"]):
                old_entry_of_row[entry["first_row"] + offset] = (entry, offset)

        # A shard whose exact row list is unchanged and contains no re-encoded row can be reused untouched.
        reusable: dict[tuple[str, ...], dict[str, Any]] = {}
        for entry in old_entries:
            row_ids = old_rows_by_shard[entry["shard"]]
            if not modified.intersection(row_ids) and not changed_vectors.keys() & set(row_ids):
                reusable.setdefault(row_ids, entry)

        total_shards = math.ceil(len(plan) / shard_size) if len(plan) else 0
        new_entries: list[dict[str, Any]] = []
        for shard in range(total_shards):
            start = shard * shard_size
            end = min(start + shard_size, len(plan))
            row_ids = plan.ids[start:end]
            reused = reusable.get(row_ids)
            if reused is not None:
                entry = dict(reused)
                entry["first_row"] = start
                new_entries.append(entry)
                continue
            # Hold the shard's claim so a concurrent prune() does not remove the file between the write and the
            # manifest swap. The store lock is already held by the caller, so the claim can only conflict with a
            # prune() in another process (which holds the store lock too) and is therefore always free.
            with self._advisory_lock(self._claim_path(shard), blocking=False):
                vectors = self._materialize_rows(row_ids, old_index_by_id, old_entry_of_row, changed_vectors)
                array = self._encode_vectors(vectors)
                data = _npy_array_bytes(array)
                data_sha256 = _sha256_bytes(data)
                ids_sha256 = _hash_lines(row_ids)
                data_name, meta_name = self._shard_file_names(shard, data_sha256, ids_sha256)
                meta = {
                    "shard": shard,
                    "first_row": start,
                    "num_rows": int(array.shape[0]),
                    "dim": int(array.shape[1]),
                    "dtype": str(array.dtype),
                    "ids": list(row_ids),
                    "text_sha256": list(plan.text_hashes[start:end]),
                    "ids_sha256": ids_sha256,
                    "data_sha256": data_sha256,
                }
                _atomic_write_bytes(self.shards_path / data_name, data)
                _atomic_write_json(self.shards_path / meta_name, meta)
            new_entries.append(
                {
                    "shard": shard,
                    "first_row": start,
                    "num_rows": meta["num_rows"],
                    "dim": meta["dim"],
                    "dtype": meta["dtype"],
                    "data": f"{SHARDS_DIR}/{data_name}",
                    "meta": f"{SHARDS_DIR}/{meta_name}",
                    "data_sha256": data_sha256,
                    "ids_sha256": ids_sha256,
                }
            )

        plan_file = self._write_plan_file(plan)
        manifest = self._reload()
        if manifest["plan_hash"] != expected_plan_hash:
            raise PlanMismatchError(
                f"The corpus plan of {self.path} changed while this delta was being prepared, so it was "
                "computed against an outdated layout. Re-run apply_delta() to apply it to the current store."
            )
        manifest.update(
            num_rows=len(plan),
            num_shards=total_shards,
            plan_hash=plan.plan_hash,
            ids_hash=plan.ids_hash,
            num_duplicates_dropped=0,
            plan_file=plan_file,
            shards=new_entries,
            dim=new_entries[0]["dim"] if new_entries else None,
            dtype=new_entries[0]["dtype"] if new_entries else manifest["dtype"],
        )
        self._write_manifest(manifest)

    def _materialize_rows(
        self,
        row_ids: Sequence[str],
        old_index_by_id: dict[str, int],
        old_entry_of_row: dict[int, tuple[dict[str, Any], int]],
        changed_vectors: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build one shard's vectors from newly computed rows and memmapped old shard rows."""
        arrays: list[np.ndarray] = []
        open_mmaps: list[np.ndarray] = []
        mmap_by_file: dict[str, np.ndarray] = {}
        try:
            for row_id in row_ids:
                if row_id in changed_vectors:
                    arrays.append(np.array(changed_vectors[row_id], copy=True))
                    continue
                entry, offset = old_entry_of_row[old_index_by_id[row_id]]
                mmap = mmap_by_file.get(entry["data"])
                if mmap is None:
                    mmap = np.load(self.path / entry["data"], mmap_mode="r", allow_pickle=False)
                    mmap_by_file[entry["data"]] = mmap
                    open_mmaps.append(mmap)
                # Copy the row out before the memory map is closed, so it stays valid for the stack below.
                arrays.append(np.array(mmap[offset], copy=True))
            if not arrays:
                return np.zeros((0, 0), dtype=np.float32)
            return np.stack(arrays)
        finally:
            for mmap in open_mmaps:
                if getattr(mmap, "_mmap", None) is not None:
                    mmap._mmap.close()

    # ------------------------------------------------------------------ maintenance

    def prune(self, *, min_age_seconds: float = 300.0) -> list[str]:
        """Delete shard files the manifest no longer references, plus stale temporary files.

        Files belonging to a shard whose claim is held by another live process are left alone. So are files
        younger than ``min_age_seconds``: a concurrent delta or run writes its new shards before recording them
        in the manifest, and pruning them from under it would corrupt the store. Leftovers from a crash are
        older than the grace period and are removed on the next call.

        Args:
            min_age_seconds: Age below which unreferenced files are left untouched.

        Returns:
            The relative paths that were removed.
        """
        self._require_manifest()
        referenced = {entry["data"].split("/", 1)[-1] for entry in self.manifest["shards"]} | {
            entry["meta"].split("/", 1)[-1] for entry in self.manifest["shards"]
        }
        if self.manifest.get("plan_file"):
            referenced.add(self.manifest["plan_file"].split("/", 1)[-1])
        plan_prefix = "plan-"
        referenced_plans = {name for name in referenced if name.startswith(plan_prefix)}
        cutoff = time.time() - min_age_seconds
        removed: list[str] = []

        def is_old_enough(path: Path) -> bool:
            try:
                return path.stat().st_mtime <= cutoff
            except FileNotFoundError:
                return False

        for candidate in sorted(self.shards_path.iterdir()):
            if candidate.name in referenced or not candidate.is_file():
                continue
            if candidate.name.startswith(plan_prefix):
                # Superseded plan sidecars from an earlier completed run.
                if candidate.name not in referenced_plans and is_old_enough(candidate):
                    candidate.unlink(missing_ok=True)
                    removed.append(candidate.name)
                continue
            if candidate.name.startswith(".claim-"):
                if not is_old_enough(candidate):
                    continue
                with self._advisory_lock(candidate, blocking=False) as acquired:
                    claim_available = acquired
                if claim_available:
                    candidate.unlink(missing_ok=True)
                    removed.append(candidate.name)
                continue
            if candidate.name.startswith("."):
                continue
            if not is_old_enough(candidate):
                continue
            claim = self._claim_for_file(candidate.name)
            if claim is None:
                candidate.unlink(missing_ok=True)
                removed.append(candidate.name)
                continue
            with self._advisory_lock(self.shards_path / claim, blocking=False) as acquired:
                if acquired:
                    candidate.unlink(missing_ok=True)
                    removed.append(candidate.name)
        for tmp in self.path.glob(".*.tmp-*"):
            if tmp.is_file() and is_old_enough(tmp):
                tmp.unlink(missing_ok=True)
                removed.append(tmp.name)
        return removed

    @staticmethod
    def _claim_for_file(name: str) -> str | None:
        """Map a shard artifact name back to the claim file guarding its shard index."""
        if not name.startswith("shard-"):
            return None
        parts = name.split("-", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            return None
        return f".claim-{int(parts[1]):05d}"

    # ------------------------------------------------------------------ verification

    def verify(
        self,
        model: BaseModel | None = None,
        *,
        checksums: bool = True,
        spec: EmbeddingSpec | None = None,
    ) -> VerificationReport:
        """Check a store without re-encoding anything.

        Verifies that every expected shard is present, that each shard's file holds exactly the recorded number
        of rows and matches its SHA-256, that shard row ranges tile the store without gaps or overlaps, that no
        identifier appears twice, and — when ``model`` or ``spec`` is given — that the fingerprint matches.

        Args:
            model: Model that would extend the store. Its fingerprint is compared with the store's.
            checksums: Recompute each shard's SHA-256. Disable for a cheaper structural check.
            spec: Explicit spec to compare against instead of ``model``.

        Returns:
            A :class:`VerificationReport` whose ``differences`` describe every mismatch.
        """
        report = VerificationReport(
            path=str(self.path),
            fingerprint=None,
            expected_fingerprint=None,
            num_rows=0,
            num_shards=0,
            dim=None,
        )
        if not self.manifest_path.exists():
            report.differences.append(f"no manifest at {self.manifest_path}")
            return report

        manifest = self.manifest
        report.fingerprint = manifest["fingerprint"]
        report.num_rows = manifest["num_rows"]
        report.dim = manifest["dim"]
        report.num_shards = len(manifest["shards"])

        candidate = spec
        if candidate is None and model is not None:
            candidate = self._candidate_spec(
                model,
                prompt=None,
                prompt_name=None,
                task=None,
                normalize_embeddings=None,
                truncate_dim=None,
                precision=None,
                text_normalization=None,
            )
        if candidate is not None:
            report.expected_fingerprint = candidate.fingerprint
            if candidate.fingerprint != report.fingerprint:
                report.differences.extend(self.spec.differences(candidate))

        shard_size = manifest["shard_size"]
        expected_shards = math.ceil(manifest["num_rows"] / shard_size) if manifest["num_rows"] else 0
        if manifest["num_shards"] != expected_shards:
            report.differences.append(f"num_shards: manifest={manifest['num_shards']}, expected={expected_shards}")

        entries = {entry["shard"]: entry for entry in manifest["shards"]}
        missing: list[int] = []
        all_ids: list[str] = []
        seen_ids: set[str] = set()
        duplicate_samples: list[str] = []
        rows_accounted = 0
        dims: set[int] = set()
        for shard in range(expected_shards):
            expected_first_row = shard * shard_size
            expected_rows = min(shard_size, manifest["num_rows"] - expected_first_row)
            entry = entries.get(shard)
            if entry is None:
                missing.append(shard)
                report.shards.append(
                    ShardVerification(shard=shard, num_rows=None, first_row=None, problems=("missing",))
                )
                continue
            problems: list[str] = []
            if entry["first_row"] != expected_first_row:
                problems.append(f"first_row: recorded={entry['first_row']}, expected={expected_first_row}")
            if entry["num_rows"] != expected_rows:
                problems.append(f"num_rows: recorded={entry['num_rows']}, expected={expected_rows}")
            data_path = self.path / entry["data"]
            meta_path = self.path / entry["meta"]
            dims.add(entry["dim"])
            if not data_path.exists():
                problems.append(f"data file missing: {entry['data']}")
            if not meta_path.exists():
                problems.append(f"metadata file missing: {entry['meta']}")
            if not problems:
                meta = _read_json(meta_path)
                if meta["ids_sha256"] != entry["ids_sha256"]:
                    problems.append("ids_sha256 does not match the manifest")
                if len(meta["ids"]) != entry["num_rows"]:
                    problems.append(f"metadata lists {len(meta['ids'])} identifiers for {entry['num_rows']} rows")
                if meta["data_sha256"] != entry["data_sha256"]:
                    problems.append("data_sha256 does not match the manifest")
                if meta["num_rows"] != entry["num_rows"]:
                    problems.append("metadata row count does not match the manifest")
                try:
                    array = np.load(data_path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError, EOFError) as exc:
                    problems.append(f"could not read the data file: {exc}")
                    array = None
                if array is not None:
                    if array.shape[0] != entry["num_rows"]:
                        problems.append(f"data file holds {array.shape[0]} rows, expected {entry['num_rows']}")
                    elif array.shape[1] != entry["dim"]:
                        problems.append(f"data file has dimension {array.shape[1]}, expected {entry['dim']}")
                    for row_id in meta["ids"]:
                        if row_id in seen_ids:
                            if len(duplicate_samples) < 10:
                                duplicate_samples.append(row_id)
                        else:
                            seen_ids.add(row_id)
                    all_ids.extend(meta["ids"])
                    rows_accounted += entry["num_rows"]
                    if getattr(array, "_mmap", None) is not None:
                        array._mmap.close()
                if checksums and _sha256_bytes(data_path.read_bytes()) != entry["data_sha256"]:
                    problems.append("data file does not match its recorded SHA-256")
            report.shards.append(
                ShardVerification(
                    shard=shard,
                    num_rows=entry["num_rows"],
                    first_row=entry["first_row"],
                    problems=tuple(problems),
                )
            )

        extra_shards = sorted(set(entries) - set(range(expected_shards)))
        if missing:
            report.differences.insert(0, f"{len(missing)} of {expected_shards} shards are missing: {missing[:10]}")
        if extra_shards:
            report.differences.append(f"manifest lists shards outside the expected range: {extra_shards}")
        if rows_accounted != manifest["num_rows"]:
            report.differences.append(
                f"num_rows: manifest={manifest['num_rows']}, sum over complete shards={rows_accounted}"
            )
        if len(dims) > 1:
            report.differences.append(f"shards disagree on the embedding dimension: {sorted(dims)}")
        if duplicate_samples:
            report.differences.append(f"identifiers appear in more than one row: {duplicate_samples}")
        if all_ids and _hash_lines(all_ids) != manifest["ids_hash"]:
            report.differences.append("the stored identifier list does not match the manifest's ids_hash")
        plan_file = manifest.get("plan_file")
        if plan_file:
            plan_path = self.path / plan_file
            if not plan_path.exists():
                report.differences.append(f"the corpus plan file is missing: {plan_file}")
            else:
                plan_data = _read_json(plan_path)
                plan_ids = tuple(plan_data.get("ids", ()))
                plan_hashes = tuple(plan_data.get("text_sha256", ()))
                if len(plan_ids) != len(plan_hashes):
                    report.differences.append("the corpus plan file has mismatched identifier and text-hash counts")
                elif len(plan_ids) != manifest["num_rows"]:
                    report.differences.append(f"num_rows: manifest={manifest['num_rows']}, plan file={len(plan_ids)}")
                else:
                    # Recompute from the file's contents, so an edited plan file is caught even if its hashes
                    # were left untouched.
                    plan = CorpusPlan(plan_ids, plan_hashes)
                    if plan.ids_hash != manifest["ids_hash"]:
                        report.differences.append("the corpus plan's identifiers do not match the manifest's ids_hash")
                    if plan.plan_hash != manifest["plan_hash"]:
                        report.differences.append("the corpus plan file does not match the manifest's plan_hash")
                if all_ids and plan_ids and _hash_lines(all_ids) != _hash_lines(plan_ids):
                    report.differences.append(
                        "the identifiers recorded in the shard metadata do not match the corpus plan file"
                    )
        return report
