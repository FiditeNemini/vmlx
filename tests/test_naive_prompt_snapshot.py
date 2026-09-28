"""Native Naive N-1 capture, immutable ring/indexer state, and bounded admission."""
import importlib
from types import SimpleNamespace

import pytest
mx = pytest.importorskip('mlx.core')
pytest.importorskip('mlx_lm')


def make_model():
    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime
    register_naive_n05_flash_runtime()
    rt = importlib.import_module('mlx_lm.models.naive_n05_flash')
    mx.random.seed(928)
    model = rt.Model(rt.ModelArgs(
        vocab_size=32, hidden_size=64, num_hidden_layers=2, intermediate_size=96,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, v_head_dim=32,
        swa_num_attention_heads=2, swa_num_key_value_heads=2,
        swa_head_dim=64, swa_v_head_dim=32, index_n_heads=2, index_head_dim=64,
        partial_rotary_factor=.5, index_top_k=4, sliding_window=4,
        hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 0],
    ))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def leaves(cache):
    for entry in cache:
        if hasattr(entry, 'caches'):
            yield from leaves(entry.caches)
        else:
            yield entry


@pytest.mark.parametrize('generation_suffix', [0, 3])
def test_actual_generator_snapshot_is_n_minus_one_and_detached(monkeypatch, generation_suffix):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import clone_prompt_cache
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    model = make_model()
    tokens = list(range(1, 12))
    generator = sbg.SingleBatchGenerator(model, max_tokens=5, prefill_step_size=4)
    generator.insert([tokens], max_tokens=[5], gen_prompt_lens=[generation_suffix])
    prompt, _ = generator.next()
    snapshot = prompt[0].prompt_cache_snapshot
    assert snapshot is not None
    assert [c.offset for c in leaves(snapshot)] == [10] * 3
    before = [[x * 1 for x in c.state] for c in leaves(snapshot)]
    metadata = [c.meta_state for c in leaves(snapshot)]
    mx.eval(*[x for pair in before for x in pair])
    # Continue real decode, allowing both KV writes and SWA wrap.
    while generator._request is not None:
        generator.next()
    for c, expected, meta in zip(leaves(snapshot), before, metadata):
        assert c.meta_state == meta
        assert all(bool(mx.array_equal(x, y)) for x, y in zip(c.state, expected))
    assert snapshot[0][1].state[0].dtype == mx.float32
    assert snapshot[0][1].state[1].shape[-1] == 0
    resumed = clone_prompt_cache(snapshot)
    out = model(mx.array([[tokens[-1]]]), cache=resumed)
    mx.eval(out)
    assert int(mx.argmax(out[:, -1, :]).item()) == prompt[0].token


def test_generator_hit_at_native_boundary_does_not_replay_prefix(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import clone_prompt_cache
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    model = make_model()
    tokens = list(range(1, 12))
    cold = sbg.SingleBatchGenerator(model, max_tokens=2, prefill_step_size=4)
    cold.insert([tokens], max_tokens=[2], gen_prompt_lens=[3])
    initial, _ = cold.next()
    snapshot = clone_prompt_cache(initial[0].prompt_cache_snapshot)
    cold.remove([initial[0].uid])
    warm = sbg.SingleBatchGenerator(model, max_tokens=2, prefill_step_size=4)
    warm.insert([tokens[10:]], max_tokens=[2], caches=[snapshot],
                all_tokens=[tokens[:10]], gen_prompt_lens=[3])
    resumed, _ = warm.next()
    assert resumed[0].token == initial[0].token
    assert [c.offset for c in leaves(resumed[0].prompt_cache_snapshot)] == [10, 10, 10]
    assert warm._request.context_tokens[:len(tokens)] == tokens
    warm.remove([resumed[0].uid])


@pytest.mark.parametrize('case', ['backend', 'headroom', 'unknown-headroom', 'trailer', 'other-family'])
def test_snapshot_declines_before_copy(monkeypatch, case):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash import cache_snapshot as snap
    model = make_model()
    c = model.make_cache();mx.eval(model(mx.array([[1, 2, 3]]), cache=c))
    generator = sbg.SingleBatchGenerator(model, prompt_snapshot_max_bytes=0 if case == 'backend' else 1 << 30)
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (1 << 30, 1 << 30) if case == 'headroom' else (0, 0) if case == 'unknown-headroom' else (0, 1 << 30))
    copies = []
    def unexpected_copy(cache):
        copies.append(cache)
        raise AssertionError('copy must not be attempted')
    monkeypatch.setattr(snap, 'clone_prompt_cache', unexpected_copy)
    if case == 'other-family':model.model_type='unqualified-family'
    req = SimpleNamespace(cache=c, gen_prompt_len=-1 if case == 'trailer' else 0)
    assert generator._clone_naive_prompt_snapshot(req) is None
    assert copies == []
    if case == 'backend':assert generator.prompt_snapshot_oversize_skips == 1
    if case in ('headroom', 'unknown-headroom'):assert generator.prompt_snapshot_headroom_skips == 1


def test_rejects_mismatched_native_boundaries():
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import snapshot_size
    model = make_model();c=model.make_cache();mx.eval(model(mx.array([[1,2,3]]),cache=c))
    c[0][1].offset -= 1
    with pytest.raises(ValueError, match='unequal boundaries'):
        snapshot_size(c)


def test_rejects_snapshot_bound_to_wrong_token_key():
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import snapshot_size
    model=make_model();c=model.make_cache();mx.eval(model(mx.array([[1,2,3]]),cache=c))
    with pytest.raises(ValueError,match='token key'):
        snapshot_size(c,expected_tokens=2)
    assert snapshot_size(c,expected_tokens=3)>0


def test_scheduler_stores_snapshot_with_full_mixed_swa_key(tmp_path, monkeypatch):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.utils import single_batch_generator as sbg

    class Tokenizer:
        clean_up_tokenization_spaces = False
        def decode(self, tokens):
            return ''.join(chr(65 + int(t)) for t in tokens)

    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    scheduler = Scheduler(make_model(), Tokenizer(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=4, enable_prefix_cache=True,
        use_paged_cache=False, enable_block_disk_cache=True,
        block_disk_cache_dir=str(tmp_path), paged_cache_block_size=4,
        max_cache_blocks=32,
    ))
    assert scheduler._mixed_attention_cache_model
    replays = []
    def unexpected_replay(*args, **kwargs):
        replays.append(args)
        raise AssertionError('a captured native boundary must not be re-prefilled')
    monkeypatch.setattr(scheduler, '_prefill_for_prompt_only_cache', unexpected_replay)
    tokens = list(range(1, 12))
    request = Request(request_id='snapshot-key', prompt=tokens,
                      sampling_params=SamplingParams(max_tokens=2, temperature=0))
    request.prompt_token_ids=tokens
    request.num_prompt_tokens=len(tokens)
    request._gen_prompt_len=3
    try:
        scheduler.add_request(request)
        for _ in range(5):
            result=scheduler.step()
            if any(o.finished for o in result.outputs):break
        else:pytest.fail('request did not finish')
        assert replays == []
        assert request._extracted_cache_key_tokens == tokens[:-1]
        table, remainder=scheduler.block_aware_cache.fetch_cache('snapshot-refault', tokens)
        assert table is not None and table.num_tokens == len(tokens)-1
        assert remainder == tokens[-1:]
        restored=scheduler.block_aware_cache.reconstruct_cache(table)
        assert restored is not None
        assert [c.offset for c in leaves(restored)] == [10,10,10]
    finally:
        scheduler.shutdown()


def test_prefill_realizes_each_native_chunk_on_owned_stream(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    cache = model.make_cache()
    req = SimpleNamespace(cache=cache, context_tokens=[], logits_processors=[],
                          pixel_values=None, pixel_values_videos=None)
    real_eval = mx.eval
    boundaries = []
    def observed_eval(*values):
        assert mx.default_stream(mx.default_device()) == generator._stream
        boundaries.append([c.offset for c in leaves(cache)])
        return real_eval(*values)
    monkeypatch.setattr(mx, 'eval', observed_eval)
    generator._prefill(list(range(1, 11)), req)
    assert boundaries == [[4, 4, 4], [8, 8, 8], [10, 10, 10]]
    monkeypatch.setattr(mx, 'eval', real_eval)
    reference = model.make_cache()
    with generator._stream_context():
        for chunk in ([1,2,3,4], [5,6,7,8], [9,10]):
            model(mx.array([chunk]), cache=reference)
        real_eval([c.state for c in reference])
    for actual, expected in zip(leaves(cache), leaves(reference)):
        assert actual.meta_state == expected.meta_state
        assert all(bool(mx.array_equal(x, y)) for x, y in zip(actual.state, expected.state))


def test_prefill_materialization_failure_stops_before_next_chunk(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    cache = model.make_cache()
    req = SimpleNamespace(cache=cache, context_tokens=[], logits_processors=[],
                          pixel_values=None, pixel_values_videos=None)
    def fail(*values):
        raise RuntimeError('native cache evaluation failed')
    monkeypatch.setattr(mx, 'eval', fail)
    with pytest.raises(RuntimeError, match='native cache evaluation failed'):
        generator._prefill(list(range(1, 11)), req)
    assert [c.offset for c in leaves(cache)] == [4, 4, 4]
    assert req.context_tokens == []
