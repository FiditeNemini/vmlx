"""Qwen3.5/3.8 VLM-lane proposal head: stamp-driven, env kill switch."""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from vmlx_engine import native_mtp
from vmlx_engine.native_mtp_proposal_stamp import STAMP_FILENAME
from vmlx_engine.patches.mlx_vlm_mtp.qwen35_vl import _qwen35_mtp_proposal_head


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VMLINUX_QWEN35_MTP_DRAFT_HEAD_BITS", raising=False)
    monkeypatch.delenv("VMLX_QWEN35_MTP_DRAFT_HEAD_BITS", raising=False)


def _model(head_bits: int, group_size: int = 64):
    lin = nn.Linear(256, 512, bias=False).to_quantized(
        group_size=group_size, bits=head_bits
    )
    mx.eval(lin.weight, lin.scales, lin.biases)
    return SimpleNamespace(
        lm_head=lin, args=SimpleNamespace(tie_word_embeddings=False)
    )


def _confirming_config(bundle, head_bits: int, group_size: int = 64):
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_vl",
                "tie_word_embeddings": False,
                "jang_config": {"calibrated": True},
                "quantization": {
                    "language_model.lm_head": {
                        "bits": head_bits,
                        "group_size": group_size,
                        "mode": "affine",
                    }
                },
            }
        )
    )


def test_q8_g64_head_builds_and_stamps(monkeypatch, tmp_path):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8)
    model = _model(8)
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None
    assert head.bits == 4
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["eligible"] is True
    assert stamp["family"] == "qwen3_5"
    # Second call returns the cached head without re-resolving.
    assert _qwen35_mtp_proposal_head(model) is head


def test_27b_low_bit_head_stamps_ineligible_and_uses_full_head(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 4, group_size=128)
    model = _model(4, group_size=128)
    assert _qwen35_mtp_proposal_head(model) is None
    assert model._vmlx_mtp_draft_head_state["reason"] == (
        "native_head_already_low_bit"
    )
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["eligible"] is False


def test_env_zero_kills_even_when_stamped_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    monkeypatch.setenv("VMLX_QWEN35_MTP_DRAFT_HEAD_BITS", "0")
    _confirming_config(tmp_path, 8)
    model = _model(8)
    assert _qwen35_mtp_proposal_head(model) is None
    assert model._vmlx_mtp_draft_head_state["reason"] == "disabled_by_env"
    # The one-time check still stamped the bundle for later launches.
    assert json.loads((tmp_path / STAMP_FILENAME).read_text())["eligible"] is True


def test_no_active_path_still_builds_without_stamp(monkeypatch):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", None)
    model = _model(8)
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None


def _stamp_with_tensors(bundle, source_head, bits=4, corrupt=False):
    """Write an eligible stamp pointing at calibrated tensors + the sidecar."""
    from vmlx_engine.native_mtp_proposal_stamp import TENSORS_FILENAME

    stamp = {
        "version": 1,
        "family": "qwen3_5",
        "source": {"bits": 8, "group_size": 64, "mode": "affine", "tied": False},
        "eligible": True,
        "proposal_bits": bits,
        "proposal_source": "stamped_tensors",
        "basis": "test",
    }
    (bundle / STAMP_FILENAME).write_text(json.dumps(stamp))
    dense = mx.dequantize(
        source_head.weight, source_head.scales, source_head.biases,
        group_size=source_head.group_size, bits=source_head.bits, mode="affine",
    )
    w, s, b = mx.quantize(dense, group_size=source_head.group_size, bits=bits)
    if corrupt:
        w = w[:, : int(w.shape[1]) // 2]
    mx.save_safetensors(
        str(bundle / TENSORS_FILENAME.replace(".safetensors", "")),
        {"weight": w, "scales": s, "biases": b},
    )


def test_stamped_tensors_are_loaded_when_stamp_points_at_them(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8)
    model = _model(8)
    _stamp_with_tensors(tmp_path, model.lm_head)
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None
    assert head.bits == 4
    assert model._vmlx_mtp_draft_head_state["reason"] == "ready_stamped"


def test_corrupt_stamped_tensors_fall_open_to_runtime_requant(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8)
    model = _model(8)
    _stamp_with_tensors(tmp_path, model.lm_head, corrupt=True)
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None  # still built — via RTN fallback
    assert model._vmlx_mtp_draft_head_state["reason"] == "ready"


def test_missing_tensors_file_falls_open_to_runtime_requant(
    monkeypatch, tmp_path
):
    from vmlx_engine.native_mtp_proposal_stamp import TENSORS_FILENAME

    monkeypatch.setattr(native_mtp, "_ACTIVE_NATIVE_MTP_MODEL_PATH", tmp_path)
    _confirming_config(tmp_path, 8)
    model = _model(8)
    _stamp_with_tensors(tmp_path, model.lm_head)
    (tmp_path / TENSORS_FILENAME).unlink()
    head = _qwen35_mtp_proposal_head(model)
    assert head is not None
    assert model._vmlx_mtp_draft_head_state["reason"] == "ready"
