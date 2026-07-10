# SPDX-License-Identifier: Apache-2.0
"""Hy3 (hy_v3) native MTP — autodetect + model-side contract pins.

The Hy3 JANG converter (jang_tools.convert_hy3_jang, profile JANG_2L with
mtp_policy=preserve-affine8) re-namespaces the DSV3-style MTP layer
(model.layers.80.*) to mtp.0.* with final param names. jang_tools.hy3.model
supplies Hy3MTPLayer + the vmlx batch_generator hooks (mtp_forward /
make_mtp_cache / return_hidden __call__); patches/mlx_lm_mtp/hy_v3_model.py
gates head attachment on is_mtp_active().
"""

from __future__ import annotations

import json

import pytest


def _write_hy3_jang_2l_native_bundle(path):
    """Mirror the stamps jang_tools.convert_hy3_jang writes for JANG_2L +
    mtp_policy=preserve-affine8."""
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hy_v3",
                "architectures": ["HYV3ForCausalLM"],
                "num_hidden_layers": 80,
                "num_nextn_predict_layers": 1,
            }
        )
    )
    (path / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "format_version": "2.0",
                "profile": "JANG_2L",
                "runtime": {
                    "bundle_has_mtp": True,
                    "mtp_layers": 1,
                    "mtp_mode": "preserved_native_candidate",
                    "mtp_num_speculative_tokens": 2,
                },
                "capabilities": {
                    "family": "hy_v3",
                    "modality": "text",
                    "cache_type": "kv",
                },
            }
        )
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                    "mtp.0.eh_proj.weight": "model-00001.safetensors",
                    "mtp.0.eh_proj.scales": "model-00001.safetensors",
                    "mtp.0.enorm.weight": "model-00001.safetensors",
                    "mtp.0.hnorm.weight": "model-00001.safetensors",
                    "mtp.0.final_layernorm.weight": "model-00001.safetensors",
                    "mtp.0.block.self_attn.q_proj.weight": "model-00001.safetensors",
                    "mtp.0.block.mlp.switch_mlp.gate_proj.weight": "model-00001.safetensors",
                }
            }
        )
    )


class TestHy3NativeMtpAutodetect:
    def test_jang_2l_native_bundle_is_native_runtime_ready(self, tmp_path, monkeypatch):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.delenv("VMLINUX_NATIVE_MTP", raising=False)
        _write_hy3_jang_2l_native_bundle(tmp_path)

        status = inspect_native_mtp_bundle(str(tmp_path))

        assert status["family"] == "hy_v3"
        assert status["config_num_nextn_predict_layers"] == 1
        assert status["index_has_mtp_tensors"] is True
        assert status["index_mtp_layer_count"] == 1
        assert status["jang_drop_mtp"] is not True
        assert status["issues"] == []
        assert status["artifact_available"] is True
        assert status["runtime_supported"] is True
        assert status["runtime_available"] is True
        assert status["status"] == "native_runtime_ready"
        assert status["runtime_scope"] == "text"

    def test_env_kill_switch_still_wins(self, tmp_path, monkeypatch):
        from vmlx_engine.native_mtp import inspect_native_mtp_bundle

        monkeypatch.setenv("VMLINUX_NATIVE_MTP", "0")
        _write_hy3_jang_2l_native_bundle(tmp_path)

        status = inspect_native_mtp_bundle(str(tmp_path))

        assert status["runtime_supported"] is True
        assert status["runtime_available"] is False
        assert status["status"] == "runtime_disabled"


def _tiny_hy3_args():
    from jang_tools.hy3.model import ModelArgs

    return ModelArgs.from_dict(
        {
            "model_type": "hy_v3",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "route_norm": True,
            "router_scaling_factor": 2.826,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
            "num_nextn_predict_layers": 1,
        }
    )


class TestHy3ModelMtpContract:
    def test_call_returns_prenorm_hidden_and_mtp_forward_shapes(self):
        import mlx.core as mx

        from jang_tools.hy3.model import Model

        model = Model(_tiny_hy3_args())
        model.attach_mtp()

        x = mx.array([[1, 2, 3]])
        logits, hidden = model(x, return_hidden=True)
        assert logits.shape == (1, 3, 128)
        assert hidden.shape == (1, 3, 64)

        mtp_cache = model.make_mtp_cache()
        assert len(mtp_cache) == 1

        draft_logits = model.mtp_forward(
            hidden[:, -1:, :], mx.array([[7]]), mtp_cache
        )
        assert draft_logits.shape == (1, 1, 128)
        # cache advanced by the 1-token draft step
        assert mtp_cache[0].offset == 1

        # second (recursive/verify-cycle) step continues the same cache
        d2, h2 = model.mtp_forward(
            hidden[:, -1:, :], mx.array([[9]]), mtp_cache, return_hidden=True
        )
        assert d2.shape == (1, 1, 128)
        assert h2.shape == (1, 1, 64)
        assert mtp_cache[0].offset == 2

    def test_sanitize_strips_mtp_without_head_and_keeps_with_head(self):
        import mlx.core as mx

        from jang_tools.hy3.model import Model
        from vmlx_engine.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active

        weights = {
            "model.norm.weight": mx.ones((64,)),
            "mtp.0.enorm.weight": mx.ones((64,)),
            "model.layers.2.eh_proj.weight": mx.ones((64, 128)),  # legacy MTP
        }

        # Another test module may have left the process-wide MTP gate on; once
        # the patch is applied, Model.__init__ auto-attaches the head when it
        # is. Pin the flag so `bare` really is head-free.
        prev = is_mtp_active()
        set_mtp_active(False)
        try:
            bare = Model(_tiny_hy3_args())
        finally:
            set_mtp_active(prev)
        out = bare.sanitize(dict(weights))
        assert "mtp.0.enorm.weight" not in out
        assert "model.layers.2.eh_proj.weight" not in out
        assert "model.norm.weight" in out

        headed = Model(_tiny_hy3_args())
        headed.attach_mtp()
        out = headed.sanitize(dict(weights))
        assert "mtp.0.enorm.weight" in out
        # legacy source-style MTP names are still dropped (never loaded)
        assert "model.layers.2.eh_proj.weight" not in out

    def test_vmlx_patch_gates_head_attachment(self):
        from vmlx_engine.patches.mlx_lm_mtp import (
            hy_v3_model,
            is_mtp_active,
            set_mtp_active,
        )

        assert hy_v3_model.apply() is True

        import sys

        hy = sys.modules["mlx_lm.models.hy_v3"]
        assert "_omlx_mtp_patched" in hy.Model.__dict__

        prev = is_mtp_active()
        try:
            set_mtp_active(False)
            assert not hasattr(hy.Model(_tiny_hy3_args()), "mtp")
            set_mtp_active(True)
            model = hy.Model(_tiny_hy3_args())
            assert hasattr(model, "mtp") and len(model.mtp) == 1
        finally:
            set_mtp_active(prev)

    def test_model_has_native_mtp_runtime_walk_finds_hy3_head(self):
        from jang_tools.hy3.model import Model
        from vmlx_engine.native_mtp import model_has_native_mtp_runtime
        from vmlx_engine.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active

        prev = is_mtp_active()
        set_mtp_active(False)  # see note above: patched __init__ reads this
        try:
            model = Model(_tiny_hy3_args())
        finally:
            set_mtp_active(prev)
        assert model_has_native_mtp_runtime(model) is False
        model.attach_mtp()
        assert model_has_native_mtp_runtime(model) is True
