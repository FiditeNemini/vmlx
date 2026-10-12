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
def _bundle(tmp_path, *, spec=True, drafter=True, model_type="minicpmv4_7", hidden=16):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": model_type, "text_config": _gdn_text_config()}))
    stamp = {"chat": {}}
    if spec:
        stamp["speculative"] = {"method": "dspark", "drafter_path": "dspark"}
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
