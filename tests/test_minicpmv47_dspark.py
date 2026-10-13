"""DSpark speculative decoding on MiniCPM-V-4.7: drafter contract, exact
verify + rollback, greedy equality with AR decoding, and serving routing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from tests.test_minicpmv47_runtime import _randomize, _text_config
from vmlx_engine.dspark import DSparkDrafter, DSparkStats, generate_ar, generate_dspark, load_drafter
from vmlx_engine.dspark.generate import choose_width
from vmlx_engine.dspark.minicpm_target import MiniCPMDSparkTarget
from vmlx_engine.models.mllm import MLXMultimodalLM, _load_dspark_runtime

PROMPT = [5, 17, 300, 4000, 12, 9, 77, 1001, 3, 8, 42]  # 11 rows: not a multiple of any chunk below


@pytest.fixture(autouse=True)
def _row_exact_on(monkeypatch):
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "1")


def _drafter_config(hidden: int = 16, taps=(0, 2, 3)) -> dict:
    vocab = _text_config(False)["vocab_size"]
    return {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "block_size": 8,
        "dflash_config": {
            "attention_mode": "gqa",
            "enable_confidence_head": True,
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "mask_token_id": 248076,
            "projector_type": "dspark",
            "target_layer_ids": list(taps),
        },
        "head_dim": 8,
        "hidden_size": hidden,
        "intermediate_size": 64,
        "layer_types": ["full_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "rope_dim": 8},
        "vocab_size": vocab,
        "draft_vocab_size": vocab,
        "markov_rank": 8,
        "enable_confidence_head": True,
    }


def _save_drafter(path, cfg: dict, seed: int = 0):
    path.mkdir(parents=True, exist_ok=True)
    model = DSparkDrafter(cfg)
    mx.random.seed(seed)
    flat = {k: (mx.random.normal(v.shape) * 0.05).astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(path / "model.safetensors"), flat)
    (path / "config.json").write_text(json.dumps(cfg))
    return path


def _gdn_text_config() -> dict:
    # The multi-row gated-delta Metal kernel needs key heads of >= 32 lanes.
    cfg = _text_config(False)
    cfg.update(linear_key_head_dim=32, linear_value_head_dim=32)
    return cfg


def _tiny_model():
    from mlx_vlm.models.minicpmv4_7 import TextConfig, VisionConfig

    from vmlx_engine.models.minicpmv4_7.config import ModelConfig
    from vmlx_engine.models.minicpmv4_7.minicpmv4_7 import Model

    config = ModelConfig.from_dict(
        {"model_type": "minicpmv4_7", "text_config": _gdn_text_config(), "vision_config": {},
         "insert_layer_id": 0, "downsample_mode": "16x"}
    )
    config.text_config = TextConfig.from_dict(_gdn_text_config())
    config.vision_config = VisionConfig.from_dict(
        {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 2,
         "image_size": 56, "patch_size": 14}
    )
    return Model(config)


def _tiny_target(seed: int = 3, eval_mode: bool = True) -> MiniCPMDSparkTarget:
    model = _tiny_model()
    _randomize(model, seed=seed)
    model.train(not eval_mode)  # eval: Metal gated-delta kernel; train: ops recurrence
    return MiniCPMDSparkTarget(model)


# ------------------------------------------------------------------ drafter
@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["dflash_config"].update(projector_type="dflash"),
        lambda c: c["dflash_config"].update(causal=True),
        lambda c: c["dflash_config"].update(markov_head_type="gated"),
        lambda c: c.update(layer_types=["sliding_attention", "full_attention"]),
        lambda c: c["rope_parameters"].update(rope_dim=4),
        lambda c: c.update(draft_vocab_size=32000),
        lambda c: (c.pop("enable_confidence_head"), c["dflash_config"].pop("enable_confidence_head")),
    ],
)
def test_drafter_rejects_unsupported_variants(mutate):
    cfg = _drafter_config()
    mutate(cfg)
    with pytest.raises(ValueError, match="DSpark drafter not supported"):
        DSparkDrafter(cfg)


def test_prequantized_drafter_equals_load_time_quantization(tmp_path):
    pytest.importorskip("jang_tools.format.aligned_safetensors")
    from vmlx_engine.dspark import write_quantized_drafter

    src = _save_drafter(tmp_path / "bf16", _drafter_config(hidden=64))
    dst = write_quantized_drafter(src, tmp_path / "q8")
    at_load = dict(tree_flatten(load_drafter(src, bits=8).parameters()))
    stored = dict(tree_flatten(load_drafter(dst).parameters()))
    assert at_load.keys() == stored.keys()
    assert all(bool(mx.array_equal(at_load[k], stored[k]).item()) for k in at_load)
    assert any(k.endswith(".scales") for k in stored)
    assert not any(k.endswith(".scales") for k in stored if k.startswith(("markov_head", "confidence_head")))
    assert json.loads((dst / "config.json").read_text())["quantization"]["bits"] == 8
    with pytest.raises(FileExistsError):
        write_quantized_drafter(src, dst)


def test_choose_width_maximizes_expected_tokens_per_cost():
    assert choose_width([0.99] * 7, 7) == 7
    assert choose_width([0.0] * 7, 7) == 1
    # A confident pair followed by a near-certain miss stops at the pair.
    assert choose_width([0.95, 0.95, 0.05, 0.95, 0.95, 0.95, 0.95], 7) == 2
    assert choose_width([0.99] * 7, 3) == 3


# ------------------------------------------------------- exact verify + rollback
@pytest.mark.parametrize("eval_mode", [True, False])
@pytest.mark.parametrize("width", [2, 5, 7])
def test_verify_rows_and_rollback_equal_decode_steps(width, eval_mode):
    target = _tiny_target(eval_mode=eval_mode)
    prompt = mx.array([PROMPT])
    rows = mx.array([[11, 22, 33, 44, 55, 66, 77][:width]])
    follow = [mx.array([500]), mx.array([501])]
    for keep in sorted({1, max(1, width // 2), width}):
        cache = target.make_cache()
        target.prefill(prompt, cache, (), None)
        logits, _, sink = target.verify(rows, cache, ())
        target.rollback(cache, sink, kept=keep, total=width)
        after = [target.decode(t, cache) for t in follow]

        ref_cache = target.make_cache()
        target.prefill(prompt, ref_cache, (), None)
        ref_rows = [target.decode(rows[0, i:i + 1], ref_cache) for i in range(width)]
        ref_cache = target.make_cache()
        target.prefill(prompt, ref_cache, (), None)
        for i in range(keep):
            target.decode(rows[0, i:i + 1], ref_cache)
        ref_after = [target.decode(t, ref_cache) for t in follow]

        assert all(bool(mx.array_equal(logits[i], ref_rows[i]).item()) for i in range(width))
        assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(after, ref_after)), keep


class _OracleDrafter(DSparkDrafter):
    """Drafts the true AR continuation, wrong at every fifth output position.

    Gives rounds that accept everything, nothing, and partial prefixes, so the
    loop's bookkeeping and the GDN rollback run on every acceptance count.
    """

    def __init__(self, cfg, reference, prompt_len):
        super().__init__(cfg)
        self.reference, self.prompt_len = reference, prompt_len

    def draft(self, block_hidden, anchor, lm_head, count, sampler=None):
        start = self.ctx_len - self.prompt_len + 1
        tokens = []
        for pos in range(start, start + count):
            tok = self.reference[pos] if pos < len(self.reference) else 7
            tokens.append(tok + 1 if pos % 5 == 4 else tok)
        return mx.array(tokens), mx.full((count,), 0.9, dtype=mx.float32)


def _oracle(tmp_path, reference) -> _OracleDrafter:
    src = _save_drafter(tmp_path / "d", _drafter_config())
    drafter = _OracleDrafter(_drafter_config(), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    return drafter


@pytest.mark.parametrize("prefill_step", [None, 4])
@pytest.mark.parametrize("fixed_width", [None, 1, 3, 7])
def test_dspark_greedy_equals_ar(tmp_path, prefill_step, fixed_width):
    target = _tiny_target()
    reference = generate_ar(target, PROMPT, 30, set(), prefill_step=prefill_step)
    drafter = _oracle(tmp_path, reference)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), prefill_step=prefill_step,
                          fixed_width=fixed_width, stats=stats)
    assert out == reference
    assert stats.tokens == len(reference)
    if fixed_width != 1:
        assert stats.accepted > 0 and stats.accepted < stats.drafted


def test_dspark_stops_at_eos_like_ar(tmp_path):
    target = _tiny_target()
    free = generate_ar(target, PROMPT, 30, set())
    eos = {free[9]}
    reference = generate_ar(target, PROMPT, 30, eos)
    drafter = _oracle(tmp_path, free)
    assert generate_dspark(target, drafter, PROMPT, 30, eos, fixed_width=7) == reference


# ------------------------------------------------------------------ serving
def _bundle(tmp_path, *, spec=True, drafter=True, model_type="minicpmv4_7", hidden=16, default=None):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type, "text_config": _gdn_text_config()}))
    stamp = {"chat": {}}
    if spec:
        stamp["speculative"] = {"method": "dspark", "drafter_path": "dspark"}
        if default is not None:
            stamp["speculative"]["default"] = default
    (root / "jang_config.json").write_text(json.dumps(stamp))
    if drafter:
        _save_drafter(root / "dspark", _drafter_config(hidden=hidden))
    return root


def test_load_dspark_runtime_contract(tmp_path, monkeypatch):
    model = _tiny_model()
    root = _bundle(tmp_path)
    runtime = _load_dspark_runtime(model, root)
    assert runtime is not None and runtime["max_draft"] == 7
    assert isinstance(runtime["target"], MiniCPMDSparkTarget)
    monkeypatch.setenv("VMLX_DSPARK", "0")
    assert _load_dspark_runtime(model, root) is None


@pytest.mark.parametrize(
    "kwargs, outcome",
    [
        ({"spec": False}, None),
        ({"model_type": "qwen3_5"}, None),
        ({"drafter": False}, FileNotFoundError),
        ({"hidden": 32}, ValueError),
    ],
)
def test_load_dspark_runtime_requires_the_declared_drafter(tmp_path, kwargs, outcome):
    root = _bundle(tmp_path, **kwargs)
    if outcome is None:
        assert _load_dspark_runtime(_tiny_model(), root) is None
    else:
        with pytest.raises(outcome):
            _load_dspark_runtime(_tiny_model(), root)


@pytest.mark.parametrize(
    "media, kwargs, routed",
    [
        ({}, {}, True),
        ({}, {"repetition_penalty": 1.0, "top_p": 0.9}, True),
        ({"images": ["x"]}, {}, False),
        ({"videos": ["x"]}, {}, False),
        ({"audio": ["x"]}, {}, False),
        ({}, {"repetition_penalty": 1.1}, False),
        ({}, {"presence_penalty": 0.5}, False),
        ({}, {"logit_bias": {1: 2.0}}, False),
        ({}, {"stop": ["\n"]}, False),
        ({}, {"thinking_budget": 64}, False),
    ],
)
def test_text_only_requests_route_to_dspark(media, kwargs, routed):
    engine = SimpleNamespace(_dspark={"loaded": True})
    args = (media.get("images", []), media.get("videos", []), media.get("audio", []), 0.0, kwargs)
    assert MLXMultimodalLM._dspark_eligible(engine, *args) is routed


def test_no_dspark_route_without_runtime_or_row_exact_verify(monkeypatch):
    assert MLXMultimodalLM._dspark_eligible(SimpleNamespace(_dspark=None), [], [], [], 0.0, {}) is False
    monkeypatch.setenv("VMLX_ROW_EXACT_VERIFY_QMV", "0")
    assert MLXMultimodalLM._dspark_eligible(SimpleNamespace(_dspark={"x": 1}), [], [], [], 0.0, {}) is False


# ------------------------------------------------------------- MoE target (35B-A3B shape, tiny)
@pytest.fixture(autouse=True)
def _stock_switchglu(monkeypatch):
    """The exactness tests compare verify rows against stock 1-row SwitchGLU decode. MiMo-V2 capability discovery
    (server._mimo_v2_runtime_module, only for model_type mimo_v2) installs a process-global SwitchGLU decode fast
    path that other tests leave in place; a MiniCPM serving process never has it, so run these on the stock call."""
    from mlx_lm.models.switch_layers import SwitchGLU

    original = getattr(SwitchGLU, "_vmlx_mimo_original_affine_decode_call", None)
    if original is not None:
        monkeypatch.setattr(SwitchGLU, "__call__", original)
        monkeypatch.setattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_path", False)
    yield


def _moe_text_config() -> dict:
    # Quantizable expert shapes (64) and 16 experts x top-8: an 8-row verify reaches SwitchGLU's 64-index sort
    # threshold, where the stock multi-row block switches kernels (the case the per-row MoE verify exists for).
    cfg = _text_config(True)
    cfg.update(hidden_size=64, intermediate_size=64, moe_intermediate_size=64, shared_expert_intermediate_size=64,
               num_experts=16, num_experts_per_tok=8, linear_key_head_dim=32, linear_value_head_dim=32)
    return cfg


def _tiny_moe_target(seed: int = 5, eval_mode: bool = True) -> MiniCPMDSparkTarget:
    import mlx.nn as nn
    from mlx_vlm.models.minicpmv4_7 import TextConfig, VisionConfig

    from vmlx_engine.models.minicpmv4_7.config import ModelConfig
    from vmlx_engine.models.minicpmv4_7.minicpmv4_7 import Model

    config = ModelConfig.from_dict(
        {"model_type": "minicpmv4_7", "text_config": _moe_text_config(), "vision_config": {},
         "insert_layer_id": 0, "downsample_mode": "16x"}
    )
    config.text_config = TextConfig.from_dict(_moe_text_config())
    config.vision_config = VisionConfig.from_dict(
        {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 2,
         "image_size": 56, "patch_size": 14}
    )
    model = Model(config)
    _randomize(model, seed=seed)
    # Routed experts affine 8 (the bundles' JANG_8M layout); routers stay unquantized like the bundles' bf16 routers.
    nn.quantize(model.language_model, group_size=64, bits=8,
                class_predicate=lambda path, m: "switch_mlp" in path and hasattr(m, "to_quantized"))
    model.set_dtype(mx.bfloat16)
    model.train(not eval_mode)
    mx.eval(model.parameters())
    return MiniCPMDSparkTarget(model)


def _verify_vs_decode(target, width):
    prompt = mx.array([PROMPT])
    rows = mx.array([[11, 22, 33, 44, 55, 66, 77, 88][:width]])
    cache = target.make_cache()
    target.prefill(prompt, cache, (), None)
    logits, _, sink = target.verify(rows, cache, ())
    ref_cache = target.make_cache()
    target.prefill(prompt, ref_cache, (), None)
    ref_rows = [target.decode(rows[0, i:i + 1], ref_cache) for i in range(width)]
    return [bool(mx.array_equal(logits[i], ref_rows[i]).item()) for i in range(width)], cache, sink, rows


@pytest.mark.parametrize("eval_mode", [True, False])
@pytest.mark.parametrize("width", [2, 5, 7, 8])
def test_moe_verify_rows_and_rollback_equal_decode_steps(width, eval_mode):
    target = _tiny_moe_target(eval_mode=eval_mode)
    assert target.moe
    follow = [mx.array([500]), mx.array([501])]
    for keep in sorted({1, max(1, width // 2), width}):
        same, cache, sink, rows = _verify_vs_decode(target, width)
        assert all(same), same
        target.rollback(cache, sink, kept=keep, total=width)
        after = [target.decode(t, cache) for t in follow]
        ref_cache = target.make_cache()
        target.prefill(mx.array([PROMPT]), ref_cache, (), None)
        for i in range(keep):
            target.decode(rows[0, i:i + 1], ref_cache)
        ref_after = [target.decode(t, ref_cache) for t in follow]
        assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(after, ref_after)), keep


def test_moe_control_stock_multi_row_block_is_not_row_exact_at_eight_rows():
    """Fail-closed control: the per-row MoE route is what makes 8-row verify exact on this shape."""
    target = _tiny_moe_target()
    target.moe = False                      # stock multi-row sparse MoE block inside verify
    same, *_ = _verify_vs_decode(target, 8)
    assert not all(same), "the control did not reproduce the stock block's 8-row mismatch; the MoE test would be vacuous"


@pytest.mark.parametrize("fixed_width", [None, 3, 7])
def test_moe_dspark_greedy_equals_ar(tmp_path, fixed_width):
    target = _tiny_moe_target()
    reference = generate_ar(target, PROMPT, 30, set())
    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=64))
    drafter = _OracleDrafter(_drafter_config(hidden=64), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), fixed_width=fixed_width, stats=stats)
    assert out == reference
    assert stats.accepted > 0 and stats.accepted < stats.drafted


@pytest.mark.parametrize("default, env, engaged", [
    (None, None, True), (True, None, True), (False, None, False), (False, "1", True), (True, "0", False),
])
def test_load_dspark_runtime_honours_the_stamped_default(tmp_path, monkeypatch, default, env, engaged):
    if env is None:
        monkeypatch.delenv("VMLX_DSPARK", raising=False)
    else:
        monkeypatch.setenv("VMLX_DSPARK", env)
    root = _bundle(tmp_path, default=default)
    runtime = _load_dspark_runtime(_tiny_model(), root)
    assert (runtime is not None) == engaged


def test_moe_verify_routes_below_and_at_the_expert_sort_threshold(monkeypatch):
    """Stock multi-row block below rows x top_k = 64 (one-row expert kernels, cheaper), one-row blocks at it."""
    from vmlx_engine.dspark import minicpm_target as T

    target = _tiny_moe_target()
    assert target.preferred_max_draft == 6                    # 7 rows x top-8 = 56 < 64
    block = target.nm.layers[0].mlp
    assert T._moe_rows_stay_on_decode_kernels(block, 7) and not T._moe_rows_stay_on_decode_kernels(block, 8)
    calls = []
    real = type(block).__call__

    def spy(self, x):
        calls.append(int(x.shape[1]))
        return real(self, x)
    monkeypatch.setattr(type(block), "__call__", spy)
    rows = mx.array([[11, 22, 33, 44, 55, 66, 77, 88]])
    for width, expected_calls in ((7, [7]), (8, [1] * 8)):
        cache = target.make_cache()
        target.prefill(mx.array([PROMPT]), cache, (), None)
        calls.clear()
        logits, _, _ = target.verify(rows[:, :width], cache, ())
        mx.eval(logits)
        assert calls == expected_calls * len(target.nm.layers), (width, calls[:10])
    same7, *_ = _verify_vs_decode(target, 7)
    same8, *_ = _verify_vs_decode(target, 8)
    assert all(same7) and all(same8)


# ------------------------------------------------------------- AR fallback (measured-cost policy)
def test_ar_step_equals_decode_and_returns_taps():
    for make in (_tiny_target, _tiny_moe_target):
        target = make()
        taps = (0, 2, 3)
        c1, c2 = target.make_cache(), target.make_cache()
        target.prefill(mx.array([PROMPT]), c1, taps, None)
        target.prefill(mx.array([PROMPT]), c2, taps, None)
        for tok in (500, 501, 502):
            a, feats = target.ar_step(mx.array([tok]), c1, taps)
            b = target.decode(mx.array([tok]), c2)
            assert bool(mx.array_equal(a, b).item()) and feats.shape[:2] == (1, 1)


class _ScriptedFallback:
    """Forces a fixed action sequence (True = AR step) to exercise every interleaving."""

    def __init__(self, plan):
        self.plan, self.i = list(plan), 0

    def next_is_ar(self):
        out = self.plan[self.i % len(self.plan)]
        self.i += 1
        return out

    def record_draft(self, tokens, seconds):
        pass

    def record_ar(self, seconds):
        pass

    def burst_size(self):
        return 3

    def record_ar_burst(self, steps, seconds):
        pass


@pytest.mark.parametrize("plan", [[False, True], [True, True, False], [False, False, True, True, True]])
@pytest.mark.parametrize("moe", [False, True])
def test_dspark_with_ar_steps_interleaved_equals_ar(tmp_path, plan, moe):
    target = _tiny_moe_target() if moe else _tiny_target()
    hidden = 64 if moe else 16
    reference = generate_ar(target, PROMPT, 30, set())
    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=hidden))
    drafter = _OracleDrafter(_drafter_config(hidden=hidden), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    stats = DSparkStats()
    out = generate_dspark(target, drafter, PROMPT, 30, set(), stats=stats, ar_fallback=_ScriptedFallback(plan))
    assert out == reference
    assert stats.ar_tokens > 0 and stats.rounds > 0


def test_ar_fallback_policy_switches_on_measured_rates():
    from vmlx_engine.dspark.generate import ARFallback

    p = ARFallback()
    assert not p.next_is_ar()                     # first action: a draft round
    p.record_draft(tokens=2, seconds=0.030)       # ~67 tok/s drafting
    assert p.next_is_ar()                         # second action: one AR step seeds its cost
    p.record_ar(0.010)                            # 100 tok/s AR > 67 * 1.05 -> burst
    assert p.next_is_ar() and p.ar_left == p.burst
    actions = []
    for _ in range(p.probe_every + 2):            # AR steps until the probe is due, then exactly one draft round
        is_ar = p.next_is_ar()
        actions.append(is_ar)
        if is_ar:
            p.record_ar(0.010)
        else:
            p.record_draft(tokens=2, seconds=0.030)
    assert False in actions, "the policy never re-probed drafting"
    first_draft = actions.index(False)
    assert all(actions[:first_draft]) and actions[first_draft + 1] is True   # back to AR after one probe round
    q = ARFallback()
    q.record_draft(tokens=6, seconds=0.030)       # 200 tok/s drafting
    q.ar_seconds = 0.010
    assert not q.next_is_ar()                     # drafting pays: stay on draft rounds


def test_drafter_cache_context_equals_concatenated_reference(tmp_path):
    """The KVCache-backed context (append + trim of the block rows) gives the same block hidden states as attending
    over an explicit concatenation, across two add_context calls and two consecutive blocks."""
    import mlx.nn as nn

    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=16))
    drafter = load_drafter(src, bits=16)
    mx.random.seed(4)
    feats1 = mx.random.normal((1, 9, 3 * 16)).astype(mx.bfloat16)
    feats2 = mx.random.normal((1, 5, 3 * 16)).astype(mx.bfloat16)
    block = mx.random.normal((1, drafter.block, 16)).astype(mx.bfloat16)

    def reference(feature_chunks, blk):
        ks, vs, pos = [[] for _ in drafter.layers], [[] for _ in drafter.layers], 0
        for f in feature_chunks:
            ctx = drafter.hidden_norm(drafter.fc(f))
            n = int(f.shape[1])
            for li, layer in enumerate(drafter.layers):
                a = layer.self_attn
                k = a.k_norm(a.k_proj(ctx).reshape(1, n, a.kv_heads, a.head_dim)).transpose(0, 2, 1, 3)
                ks[li].append(drafter._rope(k, pos))
                vs[li].append(a.v_proj(ctx).reshape(1, n, a.kv_heads, a.head_dim).transpose(0, 2, 1, 3))
            pos += n
        h, L = blk, int(blk.shape[1])
        for li, layer in enumerate(drafter.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = drafter._rope(a.q_norm(a.q_proj(x).reshape(1, L, a.heads, a.head_dim)).transpose(0, 2, 1, 3), pos)
            k = drafter._rope(a.k_norm(a.k_proj(x).reshape(1, L, a.kv_heads, a.head_dim)).transpose(0, 2, 1, 3), pos)
            v = a.v_proj(x).reshape(1, L, a.kv_heads, a.head_dim).transpose(0, 2, 1, 3)
            k = mx.concatenate(ks[li] + [k], axis=2)
            v = mx.concatenate(vs[li] + [v], axis=2)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.head_dim**-0.5, mask=None)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, L, -1))
            y = layer.post_attention_layernorm(h)
            h = h + layer.mlp.down_proj(nn.silu(layer.mlp.gate_proj(y)) * layer.mlp.up_proj(y))
        return drafter.norm(h)

    drafter.reset()
    drafter.add_context(feats1)
    first = drafter(block)
    mx.eval(first)
    drafter.add_context(feats2)
    second = drafter(block)
    mx.eval(second)
    assert drafter.ctx_len == 14 and drafter.context_caches()[0].offset == 14
    for got, ref in ((first, reference([feats1], block)), (second, reference([feats1, feats2], block))):
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        assert diff <= 2e-2 * (mx.abs(ref.astype(mx.float32)).max().item() + 1e-6), diff


# ------------------------------------------------------------- multi-turn reuse (dspark/session.py)
def _drafter_for(target, tmp_path, reference, hidden):
    src = _save_drafter(tmp_path / "d", _drafter_config(hidden=hidden))
    drafter = _OracleDrafter(_drafter_config(hidden=hidden), reference, len(PROMPT))
    drafter.load_weights(tree_flatten(load_drafter(src, bits=16).parameters()), strict=True)
    return drafter


def _offset(target, cache):
    from vmlx_engine.dspark.minicpm_target import _cache_offset
    return _cache_offset(cache, target.nm.fa_idx)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("plan", [None, [False, True], [True, True, False]])
def test_session_state_tracks_exactly_what_the_caches_hold(tmp_path, moe, plan):
    from vmlx_engine.dspark.generate import iter_dspark
    target = _tiny_moe_target() if moe else _tiny_target()
    reference = generate_ar(target, PROMPT, 30, set())
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    eos = {reference[13]}                                     # a stop inside the run (exercises an EOS mid-burst)
    session = {}
    out = []
    for chunk in iter_dspark(target, drafter, PROMPT, 30, eos, session=session,
                             ar_fallback=_ScriptedFallback(plan) if plan else None):
        out.extend(chunk)
    fw = session["forwarded"]
    assert fw[: len(PROMPT)] == PROMPT and fw[len(PROMPT):] == out[: len(fw) - len(PROMPT)]
    assert _offset(target, session["target_cache"]) == len(fw) == drafter.ctx_len


@pytest.mark.parametrize("moe", [False, True])
def test_resume_from_a_snapshot_equals_ar_with_the_same_segments(tmp_path, moe):
    """A cold request snapshots at a cut; a second prompt sharing that prefix resumes from the snapshot and must equal
    AR run with the same prefill segmentation."""
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.dspark import session as S
    from vmlx_engine.dspark.generate import iter_dspark
    target = _tiny_moe_target() if moe else _tiny_target()
    cut = 6
    reference = generate_ar(target, PROMPT, 20, set(), prefill_cuts=[cut])
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    snaps = []
    make_draft = lambda: [KVCache() for _ in drafter.layers]
    for _ in iter_dspark(target, drafter, PROMPT, 20, set(), snapshot_cuts=[cut],
                         on_snapshot=lambda c, tc, dc: snaps.append((c, S.clone(tc, target.make_cache), S.clone(dc, make_draft)))):
        pass
    assert [c for c, *_ in snaps] == [cut] and snaps[0][1] is not None and snaps[0][2] is not None
    prompt2 = PROMPT[:cut] + [901, 902, 903, 904, 905]
    ref2 = generate_ar(target, prompt2, 20, set(), prefill_cuts=[cut])
    drafter2 = _drafter_for(target, tmp_path / "2", ref2, 64 if moe else 16)
    drafter2.prompt_len = len(prompt2)
    resume = {"target_cache": snaps[0][1], "draft_cache": snaps[0][2], "cache_len": cut}
    stats = DSparkStats()
    out = []
    for chunk in iter_dspark(target, drafter2, prompt2, 20, set(), resume=resume, stats=stats):
        out.extend(chunk)
    assert out == ref2 and stats.cached_tokens == cut and stats.prefill_tokens == len(prompt2) - cut


@pytest.mark.parametrize("moe", [False, True])
def test_resume_from_the_end_of_turn_state_equals_ar(tmp_path, moe):
    from vmlx_engine.dspark.generate import iter_dspark
    target = _tiny_moe_target() if moe else _tiny_target()
    reference = generate_ar(target, PROMPT, 12, set())
    drafter = _drafter_for(target, tmp_path, reference, 64 if moe else 16)
    session = {}
    for _ in iter_dspark(target, drafter, PROMPT, 12, set(), session=session, ar_fallback=_ScriptedFallback([False, True])):
        pass
    fw = list(session["forwarded"])
    prompt2 = fw + [777, 778, 779]
    # Reference built the way the stored cache was: the prompt prefilled, every generated token DECODED (verify rows
    # and AR steps are bit-equal to decode steps), then the new turn prefilled and greedy decode.
    ref_cache = target.make_cache()
    target.prefill(mx.array([PROMPT]), ref_cache, (), None)
    for t in fw[len(PROMPT):]:
        target.decode(mx.array([t]), ref_cache)
    logits, _ = target.prefill(mx.array([prompt2[len(fw):]]), ref_cache, (), None)
    tok = mx.argmax(logits).reshape(1)
    ref2 = [int(tok.item())]
    while len(ref2) < 15:
        tok = mx.argmax(target.decode(tok, ref_cache)).reshape(1)
        ref2.append(int(tok.item()))
    drafter2 = _drafter_for(target, tmp_path / "2", ref2, 64 if moe else 16)
    drafter2.prompt_len = len(prompt2)
    resume = {"target_cache": session["target_cache"], "draft_cache": drafter.context_caches(), "cache_len": len(fw)}
    out = []
    for chunk in iter_dspark(target, drafter2, prompt2, 15, set(), resume=resume):
        out.extend(chunk)
    assert out == ref2


def test_session_store_round_trip_ram_tier(monkeypatch):
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.dspark import session as S
    monkeypatch.setattr(S, "SSD", None)
    S.clear()
    target = _tiny_target()
    cache = target.make_cache()
    target.prefill(mx.array([PROMPT]), cache, (), None)
    entry = {"kind": "turn", "tokens": list(PROMPT), "cache_len": len(PROMPT), "target_cache": cache,
             "draft_cache": [KVCache()], "model_key": "k"}
    receipt = S.store(entry)
    assert receipt.done() and receipt.vmlx_lookup_served_from_ram
    assert S.lookup("other", PROMPT + [1], None, target.make_cache, list) is None      # model key guard
    hit = S.lookup("k", PROMPT + [1, 2], None, target.make_cache, list)
    assert hit is entry and hit["source"] == "dspark-ram"
    assert S.lookup("k", PROMPT + [1, 2], None, target.make_cache, list) is None        # ownership moved
    monkeypatch.setenv("VMLX_DSPARK_PREFIX_REUSE", "0")
    S.store(dict(entry))
    assert S.lookup("k", PROMPT + [1, 2], None, target.make_cache, list) is None
    S.clear()
