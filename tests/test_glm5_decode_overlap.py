"""GLM-5.3 (glm5_next) native caches qualify for one-token decode overlap in SingleBatchGenerator: the overlapped run
equals the synchronous run (tokens, finish reasons, logprobs, final KDA/MLA/DSA state bit-identical) through length,
stop and cancel, starting from a primed cache with DSA sparse selection active (index_topk 8 < context)."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from vmlx_engine.models.glm5_next.glm5_next import Model, ModelArgs, clone_glm5_next_layer_cache  # noqa: E402
from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator  # noqa: E402


def glm_model():
    mx.random.seed(1010)
    args = ModelArgs(
        hidden_size=128, num_hidden_layers=4, vocab_size=64, num_attention_heads=2, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=32, v_head_dim=32, index_n_heads=2, index_head_dim=32, linear_num_heads=2, linear_head_dim=32,
        n_routed_experts=8, num_experts_per_tok=8, moe_intermediate_size=64, intermediate_size=128, first_k_dense_replace=1,
        index_topk=8,
        layer_types=["linear_attention", "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"],
    )
    m = Model(args)
    for layer in m.model.layers:                      # random init leaves the KDA convs/decay at zero (see lossless test)
        att = layer.self_attn
        if hasattr(att, "q_conv1d"):
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                setattr(att, n, mx.random.normal(getattr(att, n).shape) * 0.5)
            att.A_log = mx.random.normal(att.A_log.shape) * 0.5
            att.dt_bias = mx.random.normal(att.dt_bias.shape) * 0.5
    mx.eval(m.parameters())
    return m


def test_glm_overlap_requires_exact_native_layout(monkeypatch):
    model = glm_model()
    generator = SingleBatchGenerator(model)
    cache = model.make_cache()
    request = SimpleNamespace(cache=cache, output_tokens=[])
    assert not generator._can_overlap_decode(request)
    request.output_tokens = [1]
    assert generator._can_overlap_decode(request)
    request.cache = cache[:-1] + [object()]
    assert not generator._can_overlap_decode(request)
    request.cache = cache
    cache[0].set_speculative_states([tuple(cache[0].cache)])           # a verify forward in flight
    assert not generator._can_overlap_decode(request)
    cache[0].commit_speculative()
    monkeypatch.setenv("VMLX_GLM5_DECODE_OVERLAP", "0")
    assert not generator._can_overlap_decode(request)


@pytest.mark.parametrize("mode", ["length", "stop", "cancel"])
def test_glm_overlap_matches_sync(monkeypatch, mode):
    model = glm_model()
    primed = model.make_cache()

    def prime():
        for ids in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18]):   # 18 tokens > index_topk 8
            out = model(mx.array([ids]), cache=primed)
            mx.eval(out, [c.state for c in primed])

    with ThreadPoolExecutor(max_workers=1) as owner:
        owner.submit(prime).result()

    def run(overlap, stop=None):
        generator = SingleBatchGenerator(model, max_tokens=14, stop_tokens=[] if stop is None else [stop])
        monkeypatch.setattr(generator, "_logprobs_required", lambda request: True)
        if not overlap:
            monkeypatch.setattr(generator, "_can_overlap_decode", lambda request: False)
        submits = []
        submit = generator._submit_on_stream

        def record(*values):
            submits.append(True)
            return submit(*values)

        monkeypatch.setattr(generator, "_submit_on_stream", record)
        rows = []
        with ThreadPoolExecutor(max_workers=1) as worker:
            def start():
                cache = [clone_glm5_next_layer_cache(c, copy_fn=lambda v: v * 1, copy_is_detached=True) for c in primed]
                mx.eval([c.state for c in cache])
                uid = generator.insert([[19, 20]], caches=[cache])[0]
                return cache, uid
            cache, uid = worker.submit(start).result()
            cancelled = False
            while True:
                first, rest = worker.submit(generator.next).result()
                if not first and not rest:
                    break
                rows.extend(first + rest)
                if mode == "cancel" and len(rows) >= 3 and not cancelled:
                    worker.submit(generator.remove, [uid]).result()
                    assert worker.submit(generator.next).result() == ([], [])
                    worker.submit(generator.insert, [[21, 22]], max_tokens=[5]).result()
                    cancelled = True

            def snapshot():
                values = [c.state for c in cache]
                mx.eval(values)
                copied = [(k, v * 1) for k, v in tree_flatten(values) if v is not None]
                mx.eval([v for _, v in copied])
                return copied
            state = worker.submit(snapshot).result()
        return rows, state, submits

    stop = run(False)[0][2].token if mode == "stop" else None
    expected, expected_state, _ = run(False, stop)
    actual, actual_state, submits = run(True, stop)
    assert [(r.token, r.finish_reason) for r in actual] == [(r.token, r.finish_reason) for r in expected]
    for left, right in zip(actual, expected):
        assert bool(mx.array_equal(left.logprobs, right.logprobs))
    assert len(actual_state) == len(expected_state) > 0
    for (k, v), (k2, v2) in zip(actual_state, expected_state):
        assert k == k2 and bool(mx.array_equal(v, v2)), k
    if mode != "stop":
        assert submits, "the overlapped path must submit asynchronously"
