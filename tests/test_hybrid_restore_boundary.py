"""Hybrid restore invariant: restored KV offset must equal the SSM companion
boundary, else the hit is rejected (never paired split-brained)."""

from types import SimpleNamespace

from vmlx_engine.utils.cache_extent import hybrid_kv_boundary_mismatch


def _kv(offset):
    return SimpleNamespace(offset=offset, keys=object(), values=object())


def _ssm():
    return SimpleNamespace()  # recurrent slot: no offset attribute


def test_aligned_restore_passes():
    caches = [_kv(5951), _ssm(), _kv(5951), _ssm()]
    assert hybrid_kv_boundary_mismatch(caches, 5951) is None


def test_off_by_one_kv_is_rejected_with_layer_and_offset():
    caches = [_kv(5952), _ssm(), _kv(5952)]
    assert hybrid_kv_boundary_mismatch(caches, 5951) == (0, 5952)


def test_single_divergent_layer_is_caught():
    caches = [_kv(1000), _ssm(), _kv(999), _kv(1000)]
    assert hybrid_kv_boundary_mismatch(caches, 1000) == (2, 999)


def test_unknown_offsets_are_not_mismatches():
    # Recurrent slots and wrappers that never populate offset report 0.
    caches = [_ssm(), SimpleNamespace(offset=0), _ssm()]
    assert hybrid_kv_boundary_mismatch(caches, 4096) is None


def test_guards():
    assert hybrid_kv_boundary_mismatch([], 10) is None
    assert hybrid_kv_boundary_mismatch([_kv(5)], 0) is None
    assert hybrid_kv_boundary_mismatch([_kv(5)], "x") is None


def test_worker_validates_the_trimmed_checkpoint_not_the_original_hit(monkeypatch):
    """Nemotron live: KV hit704 trims to complete SSM512; both now align."""
    import vmlx_engine.scheduler as module
    from unittest.mock import Mock

    scheduler = module.Scheduler.__new__(module.Scheduler)
    scheduler._is_hybrid = True
    scheduler._uses_dsv4_cache = scheduler._uses_zaya_cache = False
    scheduler._ssm_state_cache = SimpleNamespace(fetch=lambda *_: None)
    scheduler._hybrid_kv_positions = [0]
    scheduler._hybrid_num_layers = 2
    scheduler.model = SimpleNamespace()
    scheduler.paged_cache_manager = SimpleNamespace(disk_only=True)
    state = _ssm()
    trimmed = SimpleNamespace(num_tokens=512, block_ids=list(range(8)))
    restored = _kv(512)
    scheduler.block_aware_cache = SimpleNamespace(
        block_size=64,
        trim_block_table=lambda *_: trimmed,
        reconstruct_cache=lambda *_: [restored],
    )
    scheduler._fetch_block_aligned_ssm_checkpoint = lambda *a, **kw: (512, [state])
    scheduler._remaining_tokens_after_cached_prefix = lambda request, n: request.prompt_token_ids[n:]
    scheduler._accept_paged_hit_credit = Mock()
    scheduler._release_unusable_paged_hit = Mock()
    monkeypatch.delenv('VMLX_DISABLE_SSM_PREFIX_RESUME', raising=False)
    monkeypatch.setattr(module, '_fix_hybrid_cache', lambda kv, *a, **kw: [kv[0], None])

    def request():
        return SimpleNamespace(request_id='trim704-to512', block_table=SimpleNamespace(num_tokens=704),
                               prompt_token_ids=list(range(864)))
    good = request()
    result = scheduler._finalize_hybrid_paged_cache_on_worker(good, [_kv(704)])
    assert result == [restored, state]
    assert good.cached_tokens == 512
    assert good.remaining_tokens == list(range(512, 864))
    assert good._cache_detail == 'block-disk+ssm'
    scheduler._release_unusable_paged_hit.assert_not_called()

    # The safety guard must still reject a genuinely misaligned restored KV.
    restored.offset = 511
    bad = request()
    assert scheduler._finalize_hybrid_paged_cache_on_worker(bad, [_kv(704)]) is None
    assert bad.cached_tokens == 0
    assert bad.remaining_tokens == bad.prompt_token_ids
    scheduler._release_unusable_paged_hit.assert_called_once_with(bad)
