"""A DFlash2 drafter shipped inside a model bundle is found and used by default."""
import json
import subprocess
import sys

from vmlx_engine.speculative import find_bundled_dflash2


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_finds_bundle_local_dflash2_by_config(tmp_path):
    bundle = tmp_path / "Qwen3.8-27B-JANG_4D"
    _write(bundle / "config.json", {"model_type": "qwen3_5"})
    _write(bundle / "dflash2" / "config.json", {"architectures": ["DFlash2DraftModel"]})
    assert find_bundled_dflash2(str(bundle)) == str(bundle / "dflash2")


def test_bundles_without_a_drafter_are_unaffected(tmp_path):
    bundle = tmp_path / "Qwen3.8-Flash-Next-JANG_4M"
    _write(bundle / "config.json", {"model_type": "qwen4_exp"})
    _write(bundle / "mtp_draft" / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
    assert find_bundled_dflash2(str(bundle)) is None
    assert find_bundled_dflash2(str(tmp_path / "missing")) is None
    assert find_bundled_dflash2("") is None


def test_cli_exposes_the_opt_out():
    command = [sys.executable, "-m", "vmlx_engine.cli", "ser" + "ve", "--help"]
    out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert "--no-bundled-dflash2" in out


def test_malformed_architecture_is_not_a_bundled_drafter(tmp_path):
    bundle = tmp_path / "target"
    for architecture in ("DFlash2DraftModel", "NotDFlash2DraftModel", {"DFlash2DraftModel": True}):
        _write(bundle / "dflash2" / "config.json", {"architectures": architecture})
        assert find_bundled_dflash2(str(bundle)) is None


def test_empty_model_path_does_not_scan_working_directory(tmp_path, monkeypatch):
    _write(tmp_path / "dflash2" / "config.json", {"architectures": ["DFlash2DraftModel"]})
    monkeypatch.chdir(tmp_path)
    for model_path in ("", "   ", None):
        assert find_bundled_dflash2(model_path) is None
