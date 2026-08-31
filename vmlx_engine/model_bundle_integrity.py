"""Header-only, cross-family model-bundle integrity preflight.

The checker never loads tensor payloads and never rewrites weight shards.  A
valid safetensors payload may begin at an address that is not naturally aligned
for its dtype; MLX materializes loader output into allocator-owned storage, so
that layout is compatible and is reported rather than "repaired".

The only automatic bundle mutation is regeneration of an unambiguous standard
``name-00001-of-000NN.safetensors`` index.  The replacement is fsync'd and
published with ``os.replace``.  Everything else either validates or fails
closed so the app can request a source-backed re-download.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import struct
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

SCHEMA = "vmlx-model-bundle-integrity-v1"
_MAX_HEADER_BYTES = 256 * 1024 * 1024
_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<number>\d{5})-of-(?P<total>\d{5})\.safetensors$"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_SIGNATURE_NAMES = {
    "config.json",
    "generation_config.json",
    "jang_config.json",
    "model_index.json",
    "processor_config.json",
    ".vmlx-downloading",
}


class BundleIntegrityError(RuntimeError):
    """A local bundle cannot be served safely without authoritative data."""


def _cache_root() -> Path:
    override = os.environ.get("VMLX_MODEL_INTEGRITY_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "vmlx-engine" / "model-integrity" / "v1"


def _relevant_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if (
            name in _SIGNATURE_NAMES
            or name.endswith(".safetensors")
            or name.endswith(".safetensors.index.json")
        ):
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _bundle_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _relevant_files(root):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b":")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stamp_paths(root: Path, cache_dir: Path) -> tuple[Path, Path]:
    key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json", cache_dir / f"{key}.lock"


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.vmlx-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleIntegrityError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"{label} must contain a JSON object")
    return value


def _shape_elements(shape: Any, *, path: Path, key: str) -> int:
    if not isinstance(shape, list):
        raise BundleIntegrityError(f"{path}: tensor {key!r} shape is not a list")
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise BundleIntegrityError(
                f"{path}: tensor {key!r} has invalid shape dimension {dimension!r}"
            )
        elements *= dimension
    return elements


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise BundleIntegrityError(f"{path}: truncated safetensors length prefix")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if header_bytes <= 0 or header_bytes > _MAX_HEADER_BYTES:
                raise BundleIntegrityError(
                    f"{path}: invalid safetensors header length {header_bytes}"
                )
            if 8 + header_bytes > size:
                raise BundleIntegrityError(
                    f"{path}: safetensors header extends beyond the file"
                )
            raw = handle.read(header_bytes)
        header = json.loads(raw)
    except BundleIntegrityError:
        raise
    except Exception as exc:
        raise BundleIntegrityError(f"{path}: safetensors header is invalid: {exc}") from exc
    if not isinstance(header, dict):
        raise BundleIntegrityError(f"{path}: safetensors header must be an object")

    # Let the pinned safetensors implementation reject format details this
    # lightweight reader should not duplicate (including newer dtype codes).
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as opened:
            authoritative_keys = set(opened.keys())
    except Exception as exc:
        raise BundleIntegrityError(f"{path}: safetensors rejected the file: {exc}") from exc

    data_start = 8 + header_bytes
    data_bytes = size - data_start
    tensors: dict[str, dict[str, Any]] = {}
    misaligned: list[dict[str, Any]] = []
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise BundleIntegrityError(f"{path}: tensor {key!r} metadata is not an object")
        dtype = entry.get("dtype")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise BundleIntegrityError(f"{path}: tensor {key!r} has no dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise BundleIntegrityError(f"{path}: tensor {key!r} has invalid data_offsets")
        start, end = offsets
        if start < 0 or end < start or end > data_bytes:
            raise BundleIntegrityError(
                f"{path}: tensor {key!r} range [{start}, {end}) exceeds payload"
            )
        elements = _shape_elements(entry.get("shape"), path=path, key=key)
        item_bytes = _DTYPE_BYTES.get(dtype.upper())
        if item_bytes is not None and end - start != elements * item_bytes:
            raise BundleIntegrityError(
                f"{path}: tensor {key!r} byte length does not match {dtype} shape"
            )
        absolute_offset = data_start + start
        if item_bytes and absolute_offset % item_bytes:
            misaligned.append(
                {
                    "key": key,
                    "dtype": dtype,
                    "absolute_offset": absolute_offset,
                    "required_alignment": item_bytes,
                }
            )
        tensors[key] = {
            "dtype": dtype,
            "shape": entry.get("shape"),
            "start": start,
            "end": end,
            "nbytes": end - start,
        }
    if set(tensors) != authoritative_keys:
        raise BundleIntegrityError(
            f"{path}: Python and safetensors header key sets disagree"
        )
    return {
        "relative_path": path.name,
        "tensors": tensors,
        "misaligned": misaligned,
    }


def _standard_shard_groups(paths: list[Path]) -> list[tuple[Path, str, list[Path]]]:
    grouped: dict[tuple[Path, str, int], dict[int, Path]] = {}
    for path in paths:
        match = _SHARD_RE.match(path.name)
        if not match:
            continue
        number = int(match.group("number"))
        total = int(match.group("total"))
        grouped.setdefault((path.parent, match.group("prefix"), total), {})[number] = path

    complete: list[tuple[Path, str, list[Path]]] = []
    for (parent, prefix, total), members in grouped.items():
        if set(members) == set(range(1, total + 1)):
            complete.append((parent, prefix, [members[number] for number in range(1, total + 1)]))
    return sorted(complete, key=lambda item: str(item[0] / item[1]))


def _validate_standard_shard_sets(paths: list[Path], root: Path) -> None:
    """Reject incomplete or internally contradictory standard shard sets."""

    grouped: dict[tuple[Path, str], dict[int, set[int]]] = {}
    for path in paths:
        match = _SHARD_RE.match(path.name)
        if not match:
            continue
        grouped.setdefault((path.parent, match.group("prefix")), {}).setdefault(
            int(match.group("total")), set()
        ).add(int(match.group("number")))

    for (parent, prefix), totals in grouped.items():
        relative = (parent / prefix).relative_to(root).as_posix()
        if len(totals) != 1:
            declared = ", ".join(str(total) for total in sorted(totals))
            raise BundleIntegrityError(
                f"{relative}: shards disagree on total count ({declared})"
            )
        total, present = next(iter(totals.items()))
        expected = set(range(1, total + 1))
        if present != expected:
            missing = sorted(expected - present)
            raise BundleIntegrityError(
                f"{relative}: incomplete shard set; missing indices {missing[:8]}"
            )


def _expected_weight_map(
    shards: list[Path], headers: dict[Path, dict[str, Any]]
) -> tuple[dict[str, str], int]:
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        for key, tensor in headers[shard]["tensors"].items():
            if key in weight_map:
                raise BundleIntegrityError(
                    f"duplicate tensor {key!r} in {weight_map[key]} and {shard.name}"
                )
            weight_map[key] = shard.name
            total_size += int(tensor["nbytes"])
    return dict(sorted(weight_map.items())), total_size


def _index_weight_map(path: Path) -> dict[str, str]:
    index = _read_json_object(path, str(path))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
    ):
        raise BundleIntegrityError(f"{path}: weight_map must map strings to shard names")
    return weight_map


def _repair_standard_indexes(
    root: Path,
    shard_paths: list[Path],
    headers: dict[Path, dict[str, Any]],
    *,
    repair: bool,
) -> tuple[list[str], list[str]]:
    repairs: list[str] = []
    warnings: list[str] = []
    for parent, prefix, shards in _standard_shard_groups(shard_paths):
        index_path = parent / f"{prefix}.safetensors.index.json"
        expected, total_size = _expected_weight_map(shards, headers)
        needs_repair = not index_path.exists()
        if index_path.exists():
            try:
                needs_repair = _index_weight_map(index_path) != expected
            except BundleIntegrityError:
                needs_repair = True
        if not needs_repair:
            continue
        relative = index_path.relative_to(root).as_posix()
        if not repair:
            if index_path.exists():
                raise BundleIntegrityError(f"{relative}: shard index is inconsistent")
            warnings.append(f"{relative}: standard shard index is missing")
            continue
        try:
            _atomic_json_write(
                index_path,
                {"metadata": {"total_size": total_size}, "weight_map": expected},
            )
        except OSError as exc:
            if index_path.exists():
                raise BundleIntegrityError(
                    f"{relative}: inconsistent index could not be atomically repaired: {exc}"
                ) from exc
            warnings.append(f"{relative}: missing index could not be created: {exc}")
            continue
        repairs.append(relative)
    return repairs, warnings


def _validate_indexes(
    root: Path, headers: dict[Path, dict[str, Any]]
) -> None:
    for index_path in sorted(root.rglob("*.safetensors.index.json")):
        weight_map = _index_weight_map(index_path)
        referenced: dict[Path, set[str]] = {}
        for key, filename in weight_map.items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise BundleIntegrityError(
                    f"{index_path}: unsafe shard path {filename!r}"
                )
            shard = index_path.parent / relative
            if shard not in headers:
                if not shard.exists():
                    raise BundleIntegrityError(
                        f"{index_path}: referenced shard {filename!r} is missing"
                    )
                raise BundleIntegrityError(
                    f"{index_path}: referenced shard {filename!r} was not validated"
                )
            if key not in headers[shard]["tensors"]:
                raise BundleIntegrityError(
                    f"{index_path}: tensor {key!r} is absent from {filename!r}"
                )
            referenced.setdefault(shard, set()).add(key)
        for shard, indexed_keys in referenced.items():
            actual_keys = set(headers[shard]["tensors"])
            if indexed_keys != actual_keys:
                missing = sorted(actual_keys - indexed_keys)
                raise BundleIntegrityError(
                    f"{index_path}: {shard.name} has unindexed tensors {missing[:4]}"
                )


def _scan_bundle(root: Path, *, repair: bool) -> dict[str, Any]:
    download_markers = sorted(root.rglob(".vmlx-downloading"))
    if download_markers:
        relative = download_markers[0].relative_to(root).as_posix()
        raise BundleIntegrityError(
            f"{root}: download is incomplete ({relative} is still present)"
        )

    for config_name in ("config.json", "model_index.json", "jang_config.json"):
        path = root / config_name
        if path.exists():
            _read_json_object(path, str(path))

    shard_paths = sorted(root.rglob("*.safetensors"))
    if not shard_paths:
        raise BundleIntegrityError(f"{root}: no safetensors weight files found")
    _validate_standard_shard_sets(shard_paths, root)
    headers = {path: _read_safetensors_header(path) for path in shard_paths}
    repairs, warnings = _repair_standard_indexes(
        root, shard_paths, headers, repair=repair
    )
    _validate_indexes(root, headers)

    misaligned_examples: list[dict[str, Any]] = []
    misaligned_count = 0
    tensor_count = 0
    for path, header in headers.items():
        tensor_count += len(header["tensors"])
        for item in header["misaligned"]:
            misaligned_count += 1
            if len(misaligned_examples) < 20:
                misaligned_examples.append(
                    {"file": path.relative_to(root).as_posix(), **item}
                )
    return {
        "status": "ok",
        "bundle": str(root),
        "shards": len(shard_paths),
        "tensors": tensor_count,
        "misaligned_tensors": misaligned_count,
        "misaligned_examples": misaligned_examples,
        "alignment_contract": "compatible_copy_on_load",
        "repairs": repairs,
        "warnings": warnings,
    }


def check_model_bundle(
    model_path: str | os.PathLike[str],
    *,
    repair: bool = True,
    use_cache: bool = True,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate one local bundle and atomically repair safe index defects."""

    root = Path(model_path).expanduser()
    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BundleIntegrityError(f"model bundle does not exist: {root}") from exc
    if not root.is_dir():
        raise BundleIntegrityError(f"model bundle is not a directory: {root}")

    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    stamp_path, lock_path = _stamp_paths(root, cache_root)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        fingerprint = _bundle_fingerprint(root)
        if use_cache and stamp_path.exists():
            try:
                cached = json.loads(stamp_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("schema") == SCHEMA
                and cached.get("bundle") == str(root)
                and cached.get("fingerprint") == fingerprint
                and cached.get("status") == "ok"
            ):
                return {**cached, "cache_hit": True}

        result = _scan_bundle(root, repair=repair)
        result.update(
            {
                "schema": SCHEMA,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "fingerprint": _bundle_fingerprint(root),
                "cache_hit": False,
            }
        )
        _atomic_json_write(stamp_path, result)
        return result


__all__ = ["BundleIntegrityError", "SCHEMA", "check_model_bundle"]
