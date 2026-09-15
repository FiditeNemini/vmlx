"""Bounded compiled DSA policy and exact attention; not full-model acceptance."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.metal import glm5_compiled_dsa_decode as candidate
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
from vmlx_engine.models.glm5_next.glm5_next import MLAAttention, ModelArgs


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(candidate, "_FAILED", False)
    monkeypatch.setattr(candidate, "_OBSERVED", False)
    monkeypatch.setattr(candidate, "_CALL_COUNT", 0)


def operands(total=4100):
    q = mx.random.normal((1, 64, 1, 512)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, total, 512)).astype(mx.bfloat16)
    ids = mx.concatenate((mx.arange(2048, dtype=mx.int32),
                          mx.arange(3, dtype=mx.int32) + (total // 4) * 4))[None, None]
    return q, k, ids, ids < total


def stock(q, k, ids, valid, past):
    owner = SimpleNamespace(scale=0.0625, gather_element_budget=MLAAttention.gather_element_budget)
    return MLAAttention._gather_absorbed_attention(owner, q, k, ids, valid, past=past)


@pytest.mark.parametrize("total", [4096, 4097, 4098, 4099, 4100])
def test_all_tail_masks_exact(total):
    if not candidate._compatible_runtime():
        pytest.skip("MLX0.32.2/M5 qualification")
    q, k, ids, valid = operands(total)
    expected = stock(q, k, ids, valid, total-1)
    with affine_moe_ar_scope():
        actual = candidate.glm5_compiled_dsa_output(q, k, ids, valid, past=total-1, scale=.0625, enabled=True)
    assert actual is not None
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)))
    assert candidate._CALL_COUNT == 1 and candidate._OBSERVED


@pytest.mark.parametrize("case", ["disabled", "non_ar", "runtime", "batch2", "prefill", "half", "keys_half",
                                  "heads", "rank", "selected", "index_dtype", "mask_dtype", "scale", "past", "failed"])
def test_unqualified_path_never_compiles(case, monkeypatch):
    q, k, ids, valid = operands()
    past, scale, enabled = 4099, .0625, True
    if case == "disabled": enabled = False
    elif case == "runtime": monkeypatch.setattr(candidate, "_compatible_runtime", lambda: False)
    elif case == "batch2": q = mx.broadcast_to(q, (2, 64, 1, 512))
    elif case == "prefill": q = mx.broadcast_to(q, (1, 64, 2, 512))
    elif case == "half": q = q.astype(mx.float16)
    elif case == "keys_half": k = k.astype(mx.float16)
    elif case == "heads": q = q[:, :32]
    elif case == "rank": q = q[..., :256]
    elif case == "selected": ids, valid = ids[..., :2048], valid[..., :2048]
    elif case == "index_dtype": ids = ids.astype(mx.uint32)
    elif case == "mask_dtype": valid = valid.astype(mx.int32)
    elif case == "scale": scale = .125
    elif case == "past": past = 4000
    elif case == "failed": monkeypatch.setattr(candidate, "_FAILED", True)
    def forbidden():
        raise AssertionError("unqualified layout reached compilation")
    monkeypatch.setattr(candidate, "_compiled_core", forbidden)
    if case == "non_ar":
        assert candidate.glm5_compiled_dsa_output(q, k, ids, valid, past=past, scale=scale, enabled=enabled) is None
    else:
        with affine_moe_ar_scope():
            assert candidate.glm5_compiled_dsa_output(q, k, ids, valid, past=past, scale=scale, enabled=enabled) is None


def test_compile_failure_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(candidate, "_compatible_runtime", lambda: True)
    def broken():
        raise RuntimeError("controlled compiler failure")
    monkeypatch.setattr(candidate, "_compiled_core", broken)
    with affine_moe_ar_scope():
        assert candidate.glm5_compiled_dsa_output(*operands(), past=4099, scale=.0625, enabled=True) is None
    assert candidate._FAILED and not candidate._OBSERVED and candidate._CALL_COUNT == 0
    assert "retaining stock attention" in caplog.text


def test_growing_history_does_not_retrace_selected_core():
    if not candidate._compatible_runtime():
        pytest.skip("MLX0.32.2/M5 qualification")
    trace_count = None
    with affine_moe_ar_scope():
        for total in (4100, 4101, 4102, 4103, 8192):
            q, k, ids, valid = operands(total)
            actual = candidate.glm5_compiled_dsa_output(q, k, ids, valid, past=total-1, scale=.0625, enabled=True)
            assert actual is not None
            expected = stock(q, k, ids, valid, total-1)
            mx.eval(expected, actual)
            assert bool(mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)))
            if trace_count is None:
                trace_count = candidate._TRACE_COUNT
            assert candidate._TRACE_COUNT == trace_count
    assert candidate._CALL_COUNT == 5


def test_default_off_and_only_glm_namespace(monkeypatch):
    from vmlx_engine.glm5_decode_policy import glm5_compiled_dsa_requested
    from vmlx_engine.prefix_cache import compute_model_cache_key
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_COMPILED_DSA_DECODE", raising=False)
        assert not glm5_compiled_dsa_requested()
        before = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_COMPILED_DSA_DECODE", "1")
        assert glm5_compiled_dsa_requested()
        after = compute_model_cache_key(model)
        assert (before != after) == family.startswith("glm5_next")


def test_model_layer_snapshots_explicit_policy(monkeypatch):
    # Small parameter allocations; the helper guards actual runtime geometry.
    args = ModelArgs(hidden_size=8, num_hidden_layers=1, num_attention_heads=1,
                     q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4,
                     v_head_dim=4, index_n_heads=1, index_head_dim=4)
    monkeypatch.delenv("VMLX_GLM5_COMPILED_DSA_DECODE", raising=False)
    assert not MLAAttention(args)._compiled_dsa_decode
    monkeypatch.setenv("VMLX_GLM5_COMPILED_DSA_DECODE", "1")
    assert MLAAttention(args)._compiled_dsa_decode
