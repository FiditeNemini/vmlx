"""Media sidecars and external runtime changes must not replay old SSD state."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmlx_engine.models.mimo_v26_contract import (
    mimo_v26_cache_identity, mimo_v26_runtime_source_identity,
)
from vmlx_engine.prefix_cache import compute_model_cache_key


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"mimo_v2"}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-1.safetensors"},
    }))
    (tmp_path / "model-1.safetensors").write_bytes(b"main")
    (tmp_path / "audio_tokenizer").mkdir()
    (tmp_path / "audio_tokenizer/config.json").write_text('{"dim":4}')
    (tmp_path / "audio_tokenizer/model.safetensors").write_bytes(b"old!")
    return tmp_path


def key(bundle, runtime="runtime-a", settings=None):
    model = SimpleNamespace(config=SimpleNamespace(model_type="mimo_v2"))
    model._vmlx_runtime_artifact_identity = mimo_v26_cache_identity(
        bundle, runtime, settings or {"audio_tokenizer_dtype": "float32"},
    )
    return compute_model_cache_key(model, str(bundle))


def test_same_bundle_reload_is_stable(bundle):
    assert key(bundle) == key(bundle)


def test_same_size_audio_weight_rewrite_invalidates_key(bundle):
    before = key(bundle)
    path = bundle / "audio_tokenizer/model.safetensors"
    stamp = path.stat().st_mtime_ns
    path.write_bytes(b"new!")
    os.utime(path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
    assert key(bundle) != before


@pytest.mark.parametrize("name", ["config.json", "audio_tokenizer/config.json"])
def test_config_content_change_with_preserved_stat_invalidates_key(bundle, name):
    before = key(bundle)
    path = bundle / name
    stamp = path.stat()
    original = path.read_text()
    replacement = original.replace("mimo_v2", "mimo_v3").replace('"dim":4', '"dim":8')
    assert len(replacement) == len(original) and replacement != original
    path.write_text(replacement)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert key(bundle) != before


def test_external_source_change_and_media_precision_invalidate_key(bundle, tmp_path):
    source = tmp_path / "runtime.py"
    source.write_text("scale = 1\n")
    runtime = mimo_v26_runtime_source_identity({"runtime": source})
    before = key(bundle, runtime)
    source.write_text("scale = 2\n")
    changed = mimo_v26_runtime_source_identity({"runtime": source})
    assert key(bundle, changed) != before
    assert key(bundle, runtime, {"audio_tokenizer_dtype": "bfloat16"}) != before


def test_removed_optional_audio_file_cannot_reuse_old_key(bundle):
    before = key(bundle)
    (bundle / "audio_tokenizer/model.safetensors").unlink()
    assert key(bundle) != before
    assert key(bundle) == key(bundle)


def test_unreadable_runtime_source_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        mimo_v26_runtime_source_identity({"runtime": tmp_path / "absent.py"})


def test_real_loader_binds_all_cache_owners_without_loading_towers(bundle, monkeypatch):
    import mlx_lm
    from vmlx_engine.models import mimo_v26 as bridge
    text = SimpleNamespace(args=SimpleNamespace(), model_type="mimo_v2")
    tokenizer = SimpleNamespace(chat_template="native")
    monkeypatch.setattr(bridge.registration_port, "register", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda path: (text, tokenizer))
    model, processor, _ = bridge.load_mimo_v26(bundle)
    identity = model._vmlx_runtime_artifact_identity
    assert len(identity) == 64
    assert model.language_model._vmlx_runtime_artifact_identity == identity
    assert text._vmlx_runtime_artifact_identity == identity
    assert processor.omni._vt is None and processor.omni._atok is None
