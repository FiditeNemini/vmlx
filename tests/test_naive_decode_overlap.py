"""Native Naive overlap must preserve sparse/ring state across SSD and stop."""
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache, load_prompt_cache, save_prompt_cache

from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime
from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator


def native_model():
    register_naive_n05_flash_runtime()
    module = importlib.import_module("mlx_lm.models.naive_n05_flash")
    mx.random.seed(930)
    args = module.ModelArgs(
        vocab_size=32, hidden_size=64, num_hidden_layers=2,
        intermediate_size=96, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, v_head_dim=32, swa_num_attention_heads=2,
        swa_num_key_value_heads=2, swa_head_dim=64, swa_v_head_dim=32,
        index_n_heads=2, index_head_dim=64, partial_rotary_factor=0.5,
        index_top_k=4, sliding_window=4,
        hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 0],
    )
    model = module.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def test_naive_overlap_requires_initialized_exact_native_layout():
    model = native_model()
    generator = SingleBatchGenerator(model)
    request = SimpleNamespace(cache=model.make_cache(), output_tokens=[])
    assert not generator._can_overlap_decode(request)
    request.output_tokens = [1]
    assert generator._can_overlap_decode(request)
    native = request.cache
    for invalid in (native[:1], native[::-1], [native[0], object()]):
        request.cache = invalid
        assert not generator._can_overlap_decode(request)
    class UnknownKV(KVCache):
        pass
    request.cache = model.make_cache()
    request.cache[0] = type(request.cache[0])(KVCache(), UnknownKV())
    assert not generator._can_overlap_decode(request)


@pytest.mark.parametrize("mode", ["length", "stop", "cancel"])
def test_naive_overlap_disk_refault_matches_sync(tmp_path, monkeypatch, mode):
    model = native_model()
    initial = model.make_cache()
    path = str(tmp_path / "native.safetensors")
    def prime():
        # Both sparse selection and SWA wrap are active before the restore.
        for ids in ([1, 2, 3], [4, 5, 6], [7]):
            value = model(mx.array([ids]), cache=initial)
            mx.eval(value, [state.state for state in initial])
        save_prompt_cache(path, initial)
    with ThreadPoolExecutor(max_workers=1) as owner:
        owner.submit(prime).result()

    def run(overlap, stop=None):
        generator = SingleBatchGenerator(
            model, max_tokens=12, stop_tokens=[] if stop is None else [stop]
        )
        monkeypatch.setattr(generator, "_logprobs_required", lambda request: True)
        if not overlap:
            monkeypatch.setattr(generator, "_can_overlap_decode", lambda request: False)
        rows = []
        submits = []
        submit = generator._submit_on_stream
        def record(*values):
            submits.append(True)
            return submit(*values)
        monkeypatch.setattr(generator, "_submit_on_stream", record)
        with ThreadPoolExecutor(max_workers=1) as worker:
            def restore():
                restored = load_prompt_cache(path)
                mx.eval([state.state for state in restored])
                uid = generator.insert([[8, 9]], caches=[restored])[0]
                return restored, uid
            restored, uid = worker.submit(restore).result()
            cancelled = False
            while True:
                first, rest = worker.submit(generator.next).result()
                if not first and not rest:
                    break
                rows.extend(first + rest)
                if mode == "cancel" and len(rows) >= 3 and not cancelled:
                    worker.submit(generator.remove, [uid]).result()
                    assert worker.submit(generator.next).result() == ([], [])
                    worker.submit(generator.insert, [[10, 11]], max_tokens=[5]).result()
                    cancelled = True
            def snapshot():
                values = [state.state for state in restored]
                mx.eval(values)
                copied = [(key, value * 1) for key, value in tree_flatten(values)]
                mx.eval([value for key, value in copied])
                return copied
            state = worker.submit(snapshot).result()
        mx.eval([value for key, value in state])
        return rows, state, [str(c.meta_state) for c in restored], submits

    stop = run(False)[0][0].token if mode == "stop" else None
    expected, expected_state, expected_meta, _ = run(False, stop)
    actual, actual_state, actual_meta, submits = run(True, stop)
    assert [(r.token, r.finish_reason) for r in actual] == [
        (r.token, r.finish_reason) for r in expected
    ]
    assert len(actual_state) == len(expected_state)
    assert actual_meta == expected_meta
    for left, right in zip(actual, expected):
        assert bool(mx.array_equal(left.logprobs, right.logprobs))
    for (key, value), (other_key, other) in zip(actual_state, expected_state):
        assert key == other_key
        assert bool(mx.array_equal(value, other))
    if mode != "stop":
        assert submits, "the actual candidate path must submit asynchronously"
