# SPDX-License-Identifier: Apache-2.0
"""Chunked vs one-shot prefill equivalence for the hybrid GDN (qwen3_5) lane.

The MLLM prefill has two text lanes for hybrid SSM models:

* ONE-SHOT (mllm_batch_generator.py ~8888): ``lm(all_uncached_tokens)`` in a
  single forward, full logits for every token.
* CHUNKED (mllm_batch_generator.py ~7920): ``lm(chunk, return_logits=False)``
  per chunk carrying KV offsets + GDN conv/ssm state through the cache, then
  one final-token forward for logits.

Hybrids defaulted to one-shot because chunked equivalence was never proven
("mask computation ... only tested for full-sequence processing").  These
tests ARE that proof at the mechanism level: they run the real vendored +
MTP-patched classes (the exact classes the serve path uses for the JANG
Qwen3.5/3.6/3.8 VLM bundles) and require BIT-IDENTICAL results between the
two lanes:

* final-token logits (what sampling sees),
* every KVCache key/value up to ``offset``,
* every GDN conv_state / ssm_state,
* a greedy continuation decoded from each cache.

Chunk grids deliberately include NON-ALIGNED sizes (a length that is not a
multiple of the chunk, chunk boundaries that do not divide the sequence) —
an aligned-only test is vacuous by construction (see the dots3
``cached_tokens % 256`` corruption).
"""

import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vmlx_engine.mllm_batch_generator import (  # noqa: E402
    _call_lm_prefix_without_logits,
    _lm_supports_return_logits,
    _materialize_prefill_cache_state,
)


def _build_lm(model_type: str, seed: int = 7):
    """Build a tiny hybrid LanguageModel from the REAL vendored+patched classes."""
    from vmlx_engine.models.qwen3_5_family import register_qwen3_5_family_runtime

    assert register_qwen3_5_family_runtime()

    from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl

    assert qwen35_vl.apply()

    if model_type == "qwen3_5":
        from mlx_vlm.models.qwen3_5.config import TextConfig
        from mlx_vlm.models.qwen3_5.language import LanguageModel

        cfg = TextConfig(
            model_type="qwen3_5_text",
            hidden_size=64,
            intermediate_size=128,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_hidden_layers=8,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=199,
            num_key_value_heads=2,
            max_position_embeddings=8192,
            head_dim=16,
            full_attention_interval=4,
        )
    elif model_type == "qwen3_5_moe":
        from mlx_vlm.models.qwen3_5_moe.config import TextConfig
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel

        cfg = TextConfig(
            model_type="qwen3_5_moe_text",
            hidden_size=64,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_hidden_layers=8,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=199,
            num_key_value_heads=2,
            max_position_embeddings=8192,
            head_dim=16,
            full_attention_interval=4,
        )
    else:  # pragma: no cover - guard for future parametrization typos
        raise ValueError(model_type)

    mx.random.seed(seed)
    lm = LanguageModel(cfg)

    # Deterministic non-trivial weights.  The default inits are fine but make
    # them explicit + bf16 to match live serve activations.
    def _randomize(m):
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_unflatten

        params = tree_flatten(m.parameters())
        new_params = []
        for name, p in params:
            new_params.append(
                (name, (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16))
            )
        m.update(tree_unflatten(new_params))
        assert isinstance(m, nn.Module)

    _randomize(lm)
    return lm


def _abs_position_ids(start: int, end: int) -> mx.array:
    pos = mx.arange(start, end, dtype=mx.int32).reshape(1, end - start)
    return mx.broadcast_to(pos[None, ...], (3, 1, end - start))


def _one_shot_prefill(lm, tokens):
    """Mirror the one-shot hybrid lane (mllm_batch_generator.py ~8888)."""
    cache = lm.make_cache()
    seq_len = tokens.shape[1]
    out = lm(tokens, cache=cache, position_ids=_abs_position_ids(0, seq_len))
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    _materialize_prefill_cache_state(cache)
    return logits[:, -1, :], cache


def _chunked_prefill(lm, tokens, chunk_sizes):
    """Mirror the chunked hybrid lane (mllm_batch_generator.py ~7920)."""
    cache = lm.make_cache()
    seq_len = tokens.shape[1]
    processed = 0
    sizes = list(chunk_sizes)
    while processed < seq_len - 1:
        step = sizes.pop(0) if sizes else (seq_len - 1 - processed)
        step = min(step, seq_len - 1 - processed)
        _call_lm_prefix_without_logits(
            lm,
            tokens[:, processed : processed + step],
            {
                "cache": cache,
                "position_ids": _abs_position_ids(processed, processed + step),
            },
        )
        _materialize_prefill_cache_state(cache)
        processed += step
    out = lm(
        tokens[:, processed:],
        cache=cache,
        position_ids=_abs_position_ids(processed, seq_len),
    )
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    _materialize_prefill_cache_state(cache)
    return logits[:, -1, :], cache


def _assert_cache_bitwise_equal(cache_a, cache_b, final_row_ulp_ok: bool = False):
    """Require bit-identical cache state between the two lanes.

    ``final_row_ulp_ok`` documents the ONE measured, bounded deviation: the
    chunked lane computes the final prompt token's K/V in a 1-token forward
    (the same shape every decode step uses), while one-shot computes it inside
    the big gemm.  MLX's shape-specialized matmul kernels may round that one
    row differently by 1 bf16 ulp (measured: max abs 1.22e-4 on one key row of
    magnitude ~0.03, all other rows and the final logits bit-identical, and
    chunked is bitwise-identical to "one-shot prefix + decode-shaped final
    token" — see test_chunked_prefill_matches_on_multiturn_suffix arm C).
    Every row except the LAST must still be bit-identical; the last row must
    be within 1 bf16 ulp of its own magnitude.
    """
    assert len(cache_a) == len(cache_b)
    for i, (a, b) in enumerate(zip(cache_a, cache_b)):
        assert type(a).__name__ == type(b).__name__, f"layer {i} cache type"
        if hasattr(a, "keys") and getattr(a, "keys", None) is not None:
            off_a = int(a.offset)
            off_b = int(b.offset)
            assert off_a == off_b, f"layer {i} offset {off_a} != {off_b}"
            for name in ("keys", "values"):
                arr_a = getattr(a, name)[..., :off_a, :]
                arr_b = getattr(b, name)[..., :off_b, :]
                if mx.array_equal(arr_a, arr_b):
                    continue
                if not final_row_ulp_ok:
                    raise AssertionError(f"layer {i} {name} differ")
                assert mx.array_equal(
                    arr_a[..., : off_a - 1, :], arr_b[..., : off_b - 1, :]
                ), f"layer {i} {name} differ before the final row"
                last_a = arr_a[..., -1:, :].astype(mx.float32)
                last_b = arr_b[..., -1:, :].astype(mx.float32)
                max_abs = float(mx.abs(last_a - last_b).max())
                row_mag = float(
                    mx.maximum(mx.abs(last_a), mx.abs(last_b)).max()
                )
                # 1 bf16 ulp at the row's magnitude bucket is 2^-8 * mag;
                # allow exactly-one-ulp at the top of the bucket via 2^-7.
                assert max_abs <= max(row_mag, 1e-6) * 2 ** -7, (
                    f"layer {i} {name} final row deviates beyond 1 bf16 ulp "
                    f"(max_abs={max_abs:.3e}, row_mag={row_mag:.3e})"
                )
        elif hasattr(a, "cache") and isinstance(getattr(a, "cache"), list):
            for j, (sa, sb) in enumerate(zip(a.cache, b.cache)):
                if sa is None or sb is None:
                    assert sa is None and sb is None, f"layer {i} state {j} None-ness"
                    continue
                assert sa.shape == sb.shape, f"layer {i} state {j} shape"
                assert mx.array_equal(sa, sb), f"layer {i} state {j} differs"


def _greedy_continue(lm, cache, last_logits, n_tokens=16):
    toks = []
    tok = mx.argmax(last_logits, axis=-1)
    for _ in range(n_tokens):
        toks.append(int(tok.item()))
        start = None
        # Absolute positions continue from the cache offset like decode does.
        for slot in cache:
            if hasattr(slot, "offset"):
                start = int(slot.offset)
                break
        out = lm(
            tok.reshape(1, 1),
            cache=cache,
            position_ids=_abs_position_ids(start, start + 1),
        )
        logits = out.logits if hasattr(out, "logits") else out
        tok = mx.argmax(logits[:, -1, :], axis=-1)
    return toks


# 331 is deliberately NOT a multiple of any chunk size used below, and the
# chunk grids include sizes that do not divide the remaining length, so chunk
# boundaries fall mid-sequence at non-aligned offsets.
_CHUNK_GRIDS = [
    pytest.param([64], id="aligned-64"),
    pytest.param([57], id="non-divisor-57"),
    pytest.param([128, 33, 5, 90], id="ragged-mixed"),
    pytest.param([1], id="token-by-token"),
    pytest.param([256, 256], id="oversize-then-tail"),
]


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("chunk_sizes", _CHUNK_GRIDS)
def test_chunked_prefill_bitwise_equals_one_shot(model_type, chunk_sizes):
    lm = _build_lm(model_type)
    assert _lm_supports_return_logits(lm), (
        "MTP patch must expose return_logits; the chunked lane depends on it"
    )
    mx.random.seed(11)
    tokens = mx.random.randint(0, 199, (1, 331))

    logits_one, cache_one = _one_shot_prefill(lm, tokens)
    logits_chunk, cache_chunk = _chunked_prefill(lm, tokens, chunk_sizes)

    if not mx.array_equal(logits_one, logits_chunk):
        diff = mx.abs(
            logits_one.astype(mx.float32) - logits_chunk.astype(mx.float32)
        )
        pytest.fail(
            "final-token logits are not bit-identical: "
            f"max|diff|={float(diff.max()):.3e} "
            f"argmax one-shot={int(mx.argmax(logits_one))} "
            f"chunked={int(mx.argmax(logits_chunk))}"
        )
    _assert_cache_bitwise_equal(cache_one, cache_chunk)

    greedy_one = _greedy_continue(lm, cache_one, logits_one)
    greedy_chunk = _greedy_continue(lm, cache_chunk, logits_chunk)
    assert greedy_one == greedy_chunk, (
        f"greedy continuations diverge: {greedy_one} vs {greedy_chunk}"
    )


@pytest.mark.parametrize("model_type", ["qwen3_5"])
def test_chunked_prefill_matches_on_multiturn_suffix(model_type):
    """A second-turn suffix prefill (non-zero starting offset) must also match.

    This is the live ten-turn shape: the cache already holds N tokens and the
    new turn's uncached suffix is prefilled on top.  The suffix length is NOT
    a multiple of the chunk size.
    """
    lm = _build_lm(model_type)
    mx.random.seed(13)
    turn1 = mx.random.randint(0, 199, (1, 173))
    turn2 = mx.random.randint(0, 199, (1, 211))

    # Arm A: turn1 one-shot, then turn2 one-shot (the current default).
    logits_a1, cache_a = _one_shot_prefill(lm, turn1)
    out_a = lm(
        turn2, cache=cache_a, position_ids=_abs_position_ids(173, 173 + 211)
    )
    logits_a = (out_a.logits if hasattr(out_a, "logits") else out_a)[:, -1, :]
    mx.eval(logits_a)

    # Arm B: turn1 one-shot (identical prefix state), then turn2 chunked.
    logits_b1, cache_b = _one_shot_prefill(lm, turn1)
    assert mx.array_equal(logits_a1, logits_b1)
    processed = 0
    seq_len = 211
    for step in (64, 64, 64, 64):
        step = min(step, seq_len - 1 - processed)
        if step <= 0:
            break
        _call_lm_prefix_without_logits(
            lm,
            turn2[:, processed : processed + step],
            {
                "cache": cache_b,
                "position_ids": _abs_position_ids(
                    173 + processed, 173 + processed + step
                ),
            },
        )
        _materialize_prefill_cache_state(cache_b)
        processed += step
    out_b = lm(
        turn2[:, processed:],
        cache=cache_b,
        position_ids=_abs_position_ids(173 + processed, 173 + seq_len),
    )
    logits_b = (out_b.logits if hasattr(out_b, "logits") else out_b)[:, -1, :]
    mx.eval(logits_b)

    assert mx.array_equal(logits_a, logits_b), (
        "suffix prefill on a warm cache is not bit-identical between lanes"
    )
    _assert_cache_bitwise_equal(cache_a, cache_b, final_row_ulp_ok=True)

    # Arm C isolates the ONLY deviation source: one-shot over the first 210
    # suffix tokens, then a decode-shaped 1-token forward for the last token.
    # Chunked (arm B) must be BITWISE identical to arm C everywhere — i.e.
    # chunking itself introduces zero deviation; the final-row ulp seen in
    # A-vs-B is purely the decode-shaped final forward, which every decode
    # step performs anyway.
    logits_c1, cache_c = _one_shot_prefill(lm, turn1)
    assert mx.array_equal(logits_a1, logits_c1)
    _call_lm_prefix_without_logits(
        lm,
        turn2[:, : seq_len - 1],
        {
            "cache": cache_c,
            "position_ids": _abs_position_ids(173, 173 + seq_len - 1),
        },
    )
    _materialize_prefill_cache_state(cache_c)
    out_c = lm(
        turn2[:, seq_len - 1 :],
        cache=cache_c,
        position_ids=_abs_position_ids(173 + seq_len - 1, 173 + seq_len),
    )
    logits_c = (out_c.logits if hasattr(out_c, "logits") else out_c)[:, -1, :]
    mx.eval(logits_c)
    assert mx.array_equal(logits_b, logits_c)
    _assert_cache_bitwise_equal(cache_b, cache_c)

    # The bounded final-row rounding must not change what gets GENERATED.
    greedy_a = _greedy_continue(lm, cache_a, logits_a, n_tokens=48)
    greedy_b = _greedy_continue(lm, cache_b, logits_b, n_tokens=48)
    assert greedy_a == greedy_b, (
        f"greedy continuations diverge after suffix prefill: "
        f"{greedy_a} vs {greedy_b}"
    )


# ---------------------------------------------------------------------------
# Path-selection gate. The per-family chunked DEFAULT was built and then
# RETRACTED on the answer-byte gate: live A/B (Qwen3.8-27B mtp16, temp 0,
# cold SSD both arms) diverged reasoning trajectories between lanes at 9.2k
# and 29k prompts while the same-lane control was byte-identical — the
# head_dim-256 SDPA fallback materializes scores with shape-dependent
# rounding, and temperature 0 amplifies one flipped near-tie into a fork.
# So: every hybrid stays ONE-SHOT by default (answer-byte parity with the
# shipped lanes), the env overrides in BOTH directions, the OOM escape hatch
# still chunks what one-shot cannot serve, and the choice is LOGGED.
# ---------------------------------------------------------------------------

import logging  # noqa: E402


def _make_gate_fixture(with_proven_config: bool):
    from mlx_lm.models.cache import KVCache

    from vmlx_engine.mllm_batch_generator import (
        MLLMBatchGenerator,
        MLLMBatchRequest,
    )

    class DummySSMCache:
        def __init__(self):
            self.state = mx.array([0])
            self.cache = [mx.array([0])]

    class DummyLanguageModel:
        def __init__(self):
            self.calls = []

        def make_cache(self):
            return [KVCache(), DummySSMCache()]

        def __call__(self, input_ids, *, return_logits=True, **kwargs):
            self.calls.append(
                {
                    "tokens": int(input_ids.shape[1]),
                    "return_logits": return_logits,
                }
            )
            return mx.zeros((input_ids.shape[0], input_ids.shape[1], 8))

    class DummyVLM:
        def __init__(self):
            self.language_model = DummyLanguageModel()
            if with_proven_config:
                self.config = {
                    "model_type": "qwen3_5",
                    "text_config": {"model_type": "qwen3_5_text"},
                }

    model = DummyVLM()
    generator = MLLMBatchGenerator(
        model=model,
        processor=object(),
        prefill_step_size=2048,
        enable_prefix_cache=False,
    )
    request = MLLMBatchRequest(
        uid=0,
        request_id="gate-fixture",
        prompt="",
        input_ids=mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32),
        temperature=0.0,
    )
    return generator, model, request


def _clear_hybrid_env(monkeypatch):
    for name in (
        "VMLX_ALLOW_HYBRID_CHUNKED_PREFILL",
        "VMLINUX_ALLOW_HYBRID_CHUNKED_PREFILL",
        "VMLX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT",
        "VMLINUX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_qwen35_fused_prefill_gate_is_narrow_and_default_on(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language

    monkeypatch.delenv("VMLINUX_QWEN35_FUSED_PREFILL", raising=False)
    monkeypatch.delenv("VMLX_QWEN35_FUSED_PREFILL", raising=False)

    q = mx.zeros((1, 24, 953, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 4, 3001, 256), dtype=mx.bfloat16)
    v = mx.zeros((1, 4, 3001, 256), dtype=mx.bfloat16)

    class KVCache:
        pass

    assert language._qwen35_force_fused_prefill(q, k, v, KVCache(), "causal")

    class TurboQuantKVCache:
        pass

    assert not language._qwen35_force_fused_prefill(
        q, k, v, TurboQuantKVCache(), "causal"
    )
    assert not language._qwen35_force_fused_prefill(q[:, :, :1], k, v, KVCache(), None)
    assert not language._qwen35_force_fused_prefill(
        q, k, v, KVCache(), mx.zeros((953, 3001), dtype=mx.bool_)
    )
    assert not language._qwen35_force_fused_prefill(
        q[..., :128], k[..., :128], v[..., :128], KVCache(), "causal"
    )

    monkeypatch.setenv("VMLX_QWEN35_FUSED_PREFILL", "0")
    assert not language._qwen35_force_fused_prefill(q, k, v, KVCache(), "causal")


def test_qwen35_fused_prefill_routes_only_eligible_attention(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language

    monkeypatch.delenv("VMLINUX_QWEN35_FUSED_PREFILL", raising=False)
    monkeypatch.delenv("VMLX_QWEN35_FUSED_PREFILL", raising=False)
    calls = []

    def fake_fused(*args, **kwargs):
        calls.append(("fused", args, kwargs))
        return mx.zeros_like(args[0])

    def fake_fallback(*args, **kwargs):
        calls.append(("fallback", args, kwargs))
        return mx.zeros_like(args[0])

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake_fused)
    monkeypatch.setattr(language, "scaled_dot_product_attention", fake_fallback)

    class KVCache:
        pass

    cache = KVCache()
    q = mx.zeros((1, 24, 9, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 4, 1024, 256), dtype=mx.bfloat16)
    v = mx.zeros((1, 4, 1024, 256), dtype=mx.bfloat16)
    language._qwen35_scaled_dot_product_attention(q, k, v, cache, 256**-0.5, "causal")
    assert calls[0][0] == "fused"
    assert calls[0][2] == {
        "scale": 256**-0.5,
        "mask": "causal",
        "force_fused": True,
    }

    calls.clear()
    language._qwen35_scaled_dot_product_attention(
        q[:, :, :1], k, v, cache, 256**-0.5, None
    )
    assert calls[0][0] == "fallback"
    assert calls[0][2] == {
        "cache": cache,
        "scale": 256**-0.5,
        "mask": None,
    }


def test_qwen_fused_d256_retires_materialized_score_oom_guard(monkeypatch):
    from vmlx_engine.mllm_batch_generator import (
        _qwen_fused_d256_owns_attention_allocation,
    )

    class Args:
        head_dim = 256

    class LanguageModel:
        args = Args()

    monkeypatch.delenv("VMLINUX_QWEN35_FUSED_PREFILL", raising=False)
    monkeypatch.delenv("VMLX_QWEN35_FUSED_PREFILL", raising=False)
    assert _qwen_fused_d256_owns_attention_allocation(
        LanguageModel(), "qwen3_5_text"
    )
    assert not _qwen_fused_d256_owns_attention_allocation(
        LanguageModel(), "minimax_m2"
    )
    monkeypatch.setenv("VMLX_QWEN35_FUSED_PREFILL", "0")
    assert not _qwen_fused_d256_owns_attention_allocation(
        LanguageModel(), "qwen3_5_text"
    )


def test_mechanism_proven_family_still_defaults_to_one_shot(monkeypatch, caplog):
    """Mechanism-level equivalence is NOT the flip gate — answer bytes are.

    The chunked default for qwen3_5 was retracted after the live A/B diverged
    (see module header). This pins the retraction so it cannot silently come
    back without a new answer-byte proof."""
    _clear_hybrid_env(monkeypatch)
    generator, model, request = _make_gate_fixture(with_proven_config=True)
    with caplog.at_level(
        logging.INFO, logger="vmlx_engine.mllm_batch_generator"
    ):
        output = generator._run_vision_encoding_inner(
            request, cache=model.language_model.make_cache()
        )
    assert output.shape == (1, 6, 8)
    assert model.language_model.calls == [
        {"tokens": 6, "return_logits": True},
    ]
    assert "Hybrid prefill path=one-shot family=qwen3_5_text" in caplog.text
    assert "fused-D256 answer-byte gate pending" in caplog.text


def test_unknown_family_defaults_to_one_shot_lane(monkeypatch, caplog):
    _clear_hybrid_env(monkeypatch)
    generator, model, request = _make_gate_fixture(with_proven_config=False)
    with caplog.at_level(
        logging.INFO, logger="vmlx_engine.mllm_batch_generator"
    ):
        output = generator._run_vision_encoding_inner(
            request, cache=model.language_model.make_cache()
        )
    assert output.shape == (1, 6, 8)
    assert model.language_model.calls == [
        {"tokens": 6, "return_logits": True},
    ]
    assert "Hybrid prefill path=one-shot family=unknown" in caplog.text


def test_tight_memory_glm_crossing_prefill_step_uses_split_oom_escape(
    monkeypatch, caplog
):
    """GLM's measured one-shot peak is not captured by heads*seq^2.

    A tight-memory GLM prompt that crosses the configured prefill step must
    reach the existing prefix-without-logits split lane.  Keep this family
    scoped: Qwen's default answer-byte gate is a separate contract.
    """
    _clear_hybrid_env(monkeypatch)
    generator, model, request = _make_gate_fixture(with_proven_config=False)
    model.config = {
        "model_type": "glm5_next",
        "text_config": {"model_type": "glm5_next_text"},
    }
    generator._tight_memory_prefill_drain = True
    request.input_ids = mx.arange(2200, dtype=mx.int32)[None, :]
    captured_boundaries = []
    monkeypatch.setattr(
        generator,
        "_ssm_capture_boundaries_for",
        lambda request, seq_len, has_images, boundary: [2199],
    )
    monkeypatch.setattr(
        generator,
        "_maybe_capture_clean_ssm_boundary",
        lambda request, cache, tokens, boundary: captured_boundaries.append(
            boundary
        ),
    )

    with caplog.at_level(
        logging.INFO, logger="vmlx_engine.mllm_batch_generator"
    ):
        output = generator._run_vision_encoding_inner(
            request, cache=model.language_model.make_cache()
        )

    assert output.shape == (1, 1, 8)
    assert model.language_model.calls == [
        {"tokens": 1024, "return_logits": False},
        {"tokens": 1024, "return_logits": False},
        {"tokens": 151, "return_logits": False},
        {"tokens": 1, "return_logits": True},
    ]
    assert captured_boundaries == [2199]
    assert "Hybrid prefill path=chunked family=glm5_next_text" in caplog.text
    assert "tight-memory GLM prefill exceeds the configured split step" in caplog.text


def test_tight_memory_glm_split_escape_respects_global_kill_switch(
    monkeypatch, caplog
):
    _clear_hybrid_env(monkeypatch)
    monkeypatch.setenv("VMLX_DISABLE_HYBRID_AUTO_CHUNK", "1")
    generator, model, request = _make_gate_fixture(with_proven_config=False)
    model.config = {
        "model_type": "glm5_next",
        "text_config": {"model_type": "glm5_next_text"},
    }
    generator._tight_memory_prefill_drain = True
    request.input_ids = mx.arange(2200, dtype=mx.int32)[None, :]

    with caplog.at_level(
        logging.INFO, logger="vmlx_engine.mllm_batch_generator"
    ):
        output = generator._run_vision_encoding_inner(
            request, cache=model.language_model.make_cache()
        )

    assert output.shape == (1, 2200, 8)
    assert model.language_model.calls == [
        {"tokens": 2200, "return_logits": True},
    ]
    assert "Hybrid prefill path=one-shot family=glm5_next_text" in caplog.text


def test_env_zero_forces_one_shot_even_for_proven_family(monkeypatch, caplog):
    _clear_hybrid_env(monkeypatch)
    monkeypatch.setenv("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL", "0")
    generator, model, request = _make_gate_fixture(with_proven_config=True)
    with caplog.at_level(
        logging.INFO, logger="vmlx_engine.mllm_batch_generator"
    ):
        output = generator._run_vision_encoding_inner(
            request, cache=model.language_model.make_cache()
        )
    assert output.shape == (1, 6, 8)
    assert model.language_model.calls == [
        {"tokens": 6, "return_logits": True},
    ]
    assert "Hybrid prefill path=one-shot family=qwen3_5_text" in caplog.text


def test_env_one_still_forces_chunked_for_unproven_family(monkeypatch):
    _clear_hybrid_env(monkeypatch)
    monkeypatch.setenv("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL", "1")
    generator, model, request = _make_gate_fixture(with_proven_config=False)
    output = generator._run_vision_encoding_inner(
        request, cache=model.language_model.make_cache()
    )
    assert output.shape == (1, 1, 8)
    assert model.language_model.calls == [
        {"tokens": 5, "return_logits": False},
        {"tokens": 1, "return_logits": True},
    ]
