from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from vmlx_engine.model_bundle_integrity import (
    BundleIntegrityError,
    check_model_bundle,
)


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for key, dtype, shape, raw in tensors:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _config(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps({"model_type": "test", "architectures": ["TestModel"]}),
        encoding="utf-8",
    )


def test_misaligned_payload_is_compatible_and_shard_is_never_rewritten(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _config(root)
    shard = root / "model.safetensors"
    _write_safetensors(
        shard,
        [
            ("pad", "U8", [1], b"\x07"),
            ("misaligned", "F32", [2], struct.pack("<2f", 1.25, -2.5)),
        ],
    )
    before = hashlib.sha256(shard.read_bytes()).hexdigest()
    cache = tmp_path / "cache"

    first = check_model_bundle(root, cache_dir=cache)
    second = check_model_bundle(root, cache_dir=cache)

    assert first["status"] == "ok"
    assert first["misaligned_tensors"] == 1
    assert first["alignment_contract"] == "compatible_copy_on_load"
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == before


def test_standard_shard_index_is_regenerated_atomically_once(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _config(root)
    _write_safetensors(
        root / "model-00001-of-00002.safetensors",
        [("a", "F32", [1], struct.pack("<f", 1.0))],
    )
    _write_safetensors(
        root / "model-00002-of-00002.safetensors",
        [("b", "F32", [1], struct.pack("<f", 2.0))],
    )
    cache = tmp_path / "cache"

    first = check_model_bundle(root, cache_dir=cache)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    second = check_model_bundle(root, cache_dir=cache)

    assert first["repairs"] == ["model.safetensors.index.json"]
    assert index["weight_map"] == {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",
    }
    assert index["metadata"]["total_size"] == 8
    assert not list(root.glob("*.tmp"))
    assert second["cache_hit"] is True
    assert second["repairs"] == []
    assert second["historical_repairs"] == ["model.safetensors.index.json"]


def test_nested_diffusers_shards_receive_their_own_atomic_index(tmp_path):
    root = tmp_path / "image-model"
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "TestPipeline"}), encoding="utf-8"
    )
    transformer = root / "transformer"
    _write_safetensors(
        transformer / "diffusion_pytorch_model-00001-of-00002.safetensors",
        [("x", "F16", [1], struct.pack("<e", 1.0))],
    )
    _write_safetensors(
        transformer / "diffusion_pytorch_model-00002-of-00002.safetensors",
        [("y", "F16", [1], struct.pack("<e", 2.0))],
    )

    result = check_model_bundle(root, cache_dir=tmp_path / "cache")

    assert result["repairs"] == [
        "transformer/diffusion_pytorch_model.safetensors.index.json"
    ]


def test_inconsistent_standard_index_is_repaired_from_complete_shards(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _config(root)
    _write_safetensors(
        root / "model-00001-of-00001.safetensors",
        [("actual", "I32", [1], struct.pack("<i", 7))],
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"wrong": "missing.safetensors"}}),
        encoding="utf-8",
    )

    result = check_model_bundle(root, cache_dir=tmp_path / "cache")

    assert result["repairs"] == ["model.safetensors.index.json"]
    repaired = json.loads((root / "model.safetensors.index.json").read_text())
    assert repaired["weight_map"] == {
        "actual": "model-00001-of-00001.safetensors"
    }


def test_missing_nonstandard_referenced_shard_fails_closed(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _config(root)
    _write_safetensors(
        root / "weights.safetensors",
        [("a", "F32", [1], struct.pack("<f", 1.0))],
    )
    (root / "custom.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(BundleIntegrityError, match="referenced shard.*is missing"):
        check_model_bundle(root, cache_dir=tmp_path / "cache")


def test_corrupt_safetensors_fails_closed(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _config(root)
    (root / "model.safetensors").write_bytes(b"not-a-valid-safetensors-file")

    with pytest.raises(BundleIntegrityError, match="safetensors"):
        check_model_bundle(root, cache_dir=tmp_path / "cache")


def test_incomplete_standard_shard_set_fails_without_an_index(tmp_path):
    root = tmp_path / "model"
    _write_safetensors(
        root / "model-00001-of-00002.safetensors",
        [("a", "F32", [1], struct.pack("<f", 1.0))],
    )

    with pytest.raises(BundleIntegrityError, match="incomplete shard set"):
        check_model_bundle(root, cache_dir=tmp_path / "cache")


def test_active_download_marker_never_receives_a_clean_stamp(tmp_path):
    root = tmp_path / "model"
    _write_safetensors(
        root / "model.safetensors",
        [("a", "F32", [1], struct.pack("<f", 1.0))],
    )
    (root / ".vmlx-downloading").write_text("active", encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="download is incomplete"):
        check_model_bundle(root, cache_dir=tmp_path / "cache")


def test_stamp_invalidates_when_bundle_metadata_changes(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _config(root)
    _write_safetensors(
        root / "model.safetensors",
        [("a", "F32", [1], struct.pack("<f", 1.0))],
    )
    cache = tmp_path / "cache"
    assert check_model_bundle(root, cache_dir=cache)["cache_hit"] is False
    assert check_model_bundle(root, cache_dir=cache)["cache_hit"] is True

    (root / "config.json").write_text(
        json.dumps({"model_type": "changed", "architectures": ["TestModel"]}),
        encoding="utf-8",
    )
    assert check_model_bundle(root, cache_dir=cache)["cache_hit"] is False


def test_direct_serve_preflight_skips_remote_ids_and_checks_local_dirs(
    tmp_path, monkeypatch
):
    import vmlx_engine.model_bundle_integrity as integrity
    from vmlx_engine import cli

    assert cli._preflight_local_model_bundle("org/remote-model") is None
    root = tmp_path / "model"
    root.mkdir()
    seen = []

    def fake_check(path, *, repair, use_cache):
        seen.append((Path(path), repair, use_cache))
        return {
            "cache_hit": True,
            "shards": 1,
            "tensors": 1,
            "misaligned_tensors": 0,
            "repairs": [],
        }

    monkeypatch.setattr(integrity, "check_model_bundle", fake_check)
    cli._preflight_local_model_bundle(str(root))
    assert seen == [(root, True, True)]


def test_bundle_check_json_does_not_initialize_runtime_or_prefix_logs(
    tmp_path, monkeypatch, capsys
):
    from vmlx_engine import cli

    root = tmp_path / "model"
    _write_safetensors(
        root / "model.safetensors",
        [("a", "F32", [1], struct.pack("<f", 1.0))],
    )
    monkeypatch.setenv("VMLX_MODEL_INTEGRITY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "_check_macos_compat", lambda: None)
    monkeypatch.setattr(cli, "_check_no_duplicate_mlx", lambda: None)
    runtime_initializations = []
    monkeypatch.setattr(
        cli,
        "_install_jangtq_wired_limit_from_sysctl",
        lambda: runtime_initializations.append(True),
    )
    monkeypatch.setattr(
        "sys.argv", ["vmlx-engine", "bundle-check", str(root), "--json"]
    )

    cli.main()

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["status"] == "ok"
    assert output.lstrip().startswith("{")
    assert runtime_initializations == []
