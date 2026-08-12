# SPDX-License-Identifier: Apache-2.0
"""nemotron_h_v2 must be indistinguishable from nemotron_h at EVERY gate.

The registry declares nemotron_h_v2 as an alias model_type of family
nemotron_h (model_configs.py), but four load-path gates matched only the
literal "nemotron_h", so a v2 bundle took a divergent load path:

  - tokenizer fallback demoted from the authoritative config.json answer to
    the registry/name-heuristic fallbacks
  - the fallback loader kept strict=True (Omni-style extra weights abort the
    load) and skipped the MoEGate quant-config sanitize (startup crash)
  - latent-MoE detection returned False, so the LatentMoE patch never
    applied — "[gather_qmm] Last dimension ... does not match" at first
    inference
  - the JANG switch_mlp fc1/fc2 rename was skipped, silently dropping expert
    weights under strict=False and running the experts on random init

Every one of those sites (plus the three that already worked) now routes
through model_configs.NEMOTRON_H_MODEL_TYPES, and these tests pin each one so
the spellings can never diverge again.
"""

from __future__ import annotations

import inspect
import json

import pytest

from vmlx_engine.model_configs import NEMOTRON_H_MODEL_TYPES


def _write_v2_config(tmp_path, extra: dict | None = None):
    config = {
        "model_type": "nemotron_h_v2",
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "num_attention_heads": 32,
        "vocab_size": 131072,
        "intermediate_size": 21504,
    }
    if extra:
        config.update(extra)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return config


def test_constant_covers_both_spellings():
    assert set(NEMOTRON_H_MODEL_TYPES) == {"nemotron_h", "nemotron_h_v2"}


def test_registry_row_derives_from_constant():
    from vmlx_engine.model_config_registry import get_model_config_registry

    rows = [
        c
        for c in get_model_config_registry()._configs
        if c.family_name == "nemotron_h"
    ]
    assert rows, "registry lost the nemotron_h family row"
    assert set(rows[0].model_types) == set(NEMOTRON_H_MODEL_TYPES), (
        "the nemotron_h registry row no longer derives from "
        "NEMOTRON_H_MODEL_TYPES — the alias list and the gates can drift again"
    )


def test_tokenizer_fallback_answers_v2_from_config_fast_path(tmp_path, monkeypatch):
    """tokenizer.py step 1 must answer for v2 without the registry.

    The registry fall-through masked this in the happy path (it re-reads
    config.json itself), so simulate the masked layer failing — exactly the
    try/except the code guards with — and the config.json fast path must
    still say True for a bundle whose directory name matches no Nemotron
    pattern.
    """
    from vmlx_engine.utils import tokenizer as tok

    bundle = tmp_path / "hybrid-56b-bundle"
    bundle.mkdir()
    _write_v2_config(bundle)

    def _boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(
        "vmlx_engine.model_config_registry.get_model_config_registry", _boom
    )

    assert tok._needs_tokenizer_fallback(str(bundle)) is True


def test_quant_sanitize_accepts_both_spellings():
    # This gate already handled v2; pinned so it stays symmetric with the rest.
    from vmlx_engine.utils.tokenizer import (
        _sanitize_nemotron_quantization_config_for_load,
    )

    for model_type in NEMOTRON_H_MODEL_TYPES:
        override, removed = _sanitize_nemotron_quantization_config_for_load(
            {
                "model_type": model_type,
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "backbone.layers.0.mixer.gate": {"group_size": 64, "bits": 8},
                },
            }
        )
        assert override is not None, f"{model_type}: gate entry not sanitized"
        assert removed == ["backbone.layers.0.mixer.gate"]


def test_fallback_loader_nonstrict_gate_uses_constant():
    """The strict=False + sanitize gate in the fallback loader must key on the
    constant. Functionally reaching it needs a real mlx_lm model load, so pin
    the source: the literal-equality form is exactly what left v2 bundles on
    strict=True (Omni-style extra weights -> hard load failure) with stale
    MoEGate quant entries intact (nn.quantize crash on MoEGate).
    """
    from vmlx_engine.utils import tokenizer as tok

    src = inspect.getsource(tok)
    assert '_cfg.get("model_type") in NEMOTRON_H_MODEL_TYPES' in src
    assert '_cfg.get("model_type") == "nemotron_h"' not in src


def test_model_inspector_latent_moe_detects_v2(tmp_path):
    from vmlx_engine.utils.model_inspector import inspect_model

    _write_v2_config(
        tmp_path,
        {
            "moe_latent_size": 1024,
            "n_routed_experts": 64,
            "num_experts_per_tok": 4,
        },
    )

    info = inspect_model(str(tmp_path))

    assert info.needs_latent_moe is True, (
        "nemotron_h_v2 with moe_latent_size must report needs_latent_moe — "
        "otherwise doctor/convert skip the LatentMoE patch and checks"
    )


def test_model_inspector_param_estimate_treats_v2_as_switch_mlp():
    from vmlx_engine.utils.model_inspector import _estimate_param_count

    base = {
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 131072,
        "intermediate_size": 21504,
    }
    v1 = _estimate_param_count({**base, "model_type": "nemotron_h"})
    v2 = _estimate_param_count({**base, "model_type": "nemotron_h_v2"})

    # Same architecture, same estimate: v2 previously fell into the 3-projection
    # SwiGLU branch and overestimated the parameter count.
    assert v1 == pytest.approx(v2)


def test_latent_moe_patch_gate_detects_v2(tmp_path):
    from vmlx_engine.utils.nemotron_latent_moe import needs_latent_moe_patch

    _write_v2_config(tmp_path, {"moe_latent_size": 1024})

    assert needs_latent_moe_patch(str(tmp_path)) is True, (
        "v2 LatentMoE bundles must get the runtime patch — without it the "
        "first inference crashes in gather_qmm on the latent/hidden mismatch"
    )


def test_jang_loader_fc_rename_gate_uses_constant():
    """The switch_mlp fc1/fc2 rename lives deep in the mmap load loop, so pin
    the source. The literal-tuple form skipped the rename for v2, and with
    strict=False the dropped up_proj/down_proj weights never error — the
    experts just run on random values.
    """
    from vmlx_engine.utils import jang_loader

    src = inspect.getsource(jang_loader)
    assert "NEMOTRON_H_MODEL_TYPES" in src
    assert '_model_type in ("nemotron_h", "nemotron")' not in src


def test_hybrid_sets_derive_from_constant():
    from vmlx_engine.utils.hybrid_tq_cache import NEMOTRON_OMNI_HYBRID_MODEL_TYPES
    from vmlx_engine.utils.ssm_companion_cache import _HYBRID_MODEL_TYPES

    assert NEMOTRON_OMNI_HYBRID_MODEL_TYPES == frozenset(NEMOTRON_H_MODEL_TYPES)
    for model_type in NEMOTRON_H_MODEL_TYPES:
        assert model_type in _HYBRID_MODEL_TYPES
