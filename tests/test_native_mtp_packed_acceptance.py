"""Packed probability readback preserves serial sampling decisions and RNG."""
import math

import mlx.core as mx
import pytest

from vmlx_engine import native_mtp_acceptance as acceptance
from vmlx_engine.sampling import make_sampler


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('reject_at', [0, 1, 2, 3])
@pytest.mark.parametrize('draw', [0., .25, .5, .75, 1.])
def test_filtered_prefix_draws_and_telemetry_match(monkeypatch, depth, reject_at, draw):
    p = [mx.log(mx.array([.4, .6])) if i == reject_at else mx.log(mx.array([.8, .2])) for i in range(depth)]
    q = [mx.log(mx.array([.8, .2])) for _ in range(depth)]
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setattr(acceptance, '_PACKED_ACCEPTANCE', enabled)
        sampler = make_sampler(temp=1., top_p=1., top_k=2)
        draws = []
        def uniform():
            draws.append(draw)
            return draw
        sampler._vmlx_random_uniform = uniform
        telemetry = {}
        count = acceptance.accepted_count([0]*depth, [0]*depth, q, p,
                    stochastic=True, sampler=sampler, filtered=True, telemetry=telemetry)
        assert telemetry.pop("packed_probability_reads", 0) == int(enabled) * (1 + int(depth > 1 and count > 0))
        outcomes.append((count, draws, telemetry))
    assert outcomes[0] == outcomes[1]
    if reject_at < depth and draw > .5:
        assert outcomes[1][0] == reject_at  # Matching target ID cannot rescue rejection.


@pytest.mark.parametrize('seed', [1, 21, 900])
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('temp,top_p', [(.6, .8), (1., .95), (1.3, 1.)])
def test_actual_seeded_sampler_residual_and_next_draw_match(monkeypatch, seed, depth, temp, top_p):
    def run(enabled):
        monkeypatch.setattr(acceptance, '_PACKED_ACCEPTANCE', enabled)
        sampler = make_sampler(temp=temp, top_p=top_p, top_k=4, seed=seed)
        ps, qs, drafts, targets = [], [], [], []
        # Includes tied cutoff values; use actual native filter and RNG hooks.
        for i in range(depth):
            p = mx.array([0., 1., 3., 3., 2., -1.]) + i*.03
            q = mx.array([3., 2., 1., 1., 0., -2.]) - i*.02
            drafts.append(int(sampler(q[None]).item()))
            targets.append(int(sampler(p[None]).item()))
            ps.append(p-mx.logsumexp(p));qs.append(q-mx.logsumexp(q))
        telemetry = {}
        count = acceptance.accepted_count(drafts, targets, qs, ps,
                    stochastic=True, sampler=sampler, telemetry=telemetry)
        correction = None
        if count < depth:
            correction, _ = acceptance.residual_sample(
                acceptance.accept_lp_for(sampler, ps[count]),
                acceptance.accept_lp_for(sampler, qs[count]), sampler=sampler)
        assert telemetry.pop("packed_probability_reads", 0) == int(enabled) * (1 + int(depth > 1 and count > 0))
        return drafts, targets, count, correction, sampler._vmlx_random_uniform(), telemetry
    assert run(False) == run(True)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_ratio_preserves_rejection(monkeypatch, value):
    sampler = make_sampler(temp=1., top_k=2)
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(acceptance, '_PACKED_ACCEPTANCE', enabled)
        results.append(acceptance.accepted_count([0], [0], [mx.array([0., -1.])],
            [mx.array([value, -1.])], stochastic=True, sampler=sampler, filtered=True))
    assert results == [0, 0]


def test_replaced_custom_filter_keeps_serial_hook_visits(monkeypatch):
    sampler = make_sampler(temp=1., top_k=2)
    visits = []
    def custom(row):
        visits.append(row)
        return row
    sampler._vmlx_acceptance_logprobs = custom
    sampler._vmlx_random_uniform = lambda: .9
    monkeypatch.setattr(acceptance, '_PACKED_ACCEPTANCE', True)
    q = mx.log(mx.array([.8, .2]));p=mx.log(mx.array([.2, .8]))
    assert acceptance.accepted_count([0, 0, 0], [0, 0, 0], [q]*3, [p]*3,
                stochastic=True, sampler=sampler) == 0
    assert len(visits) == 2  # No evaluation of unused custom-hook suffix rows.


@pytest.mark.parametrize('ids,ps,qs', [([-1], [[0.,-1.]], [[0.,-1.]]),
    ([4], [[0.,-1.]], [[0.,-1.]]), ([0], [None], [[0.,-1.]])])
def test_invalid_or_missing_rows_keep_compatibility(monkeypatch, ids, ps, qs):
    p=[mx.array(x) if x is not None else None for x in ps]
    q=[mx.array(x) if x is not None else None for x in qs]
    sampler=make_sampler(temp=1.,top_k=2)
    results=[]
    for enabled in (False,True):
        monkeypatch.setattr(acceptance,'_PACKED_ACCEPTANCE',enabled)
        results.append(acceptance.accepted_count(ids,ids,q,p,stochastic=True,sampler=sampler,filtered=True))
    assert results[0] == results[1]


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('seed', [1, 21, 900])
def test_reused_filtered_rows_preserve_correction_and_rng(monkeypatch, packed, depth, seed):
    monkeypatch.setattr(acceptance, '_PACKED_ACCEPTANCE', packed)
    outcomes = []
    for reuse in (False, True):
        sampler = make_sampler(temp=1., top_p=.95, top_k=4, seed=seed)
        p = mx.array([0., 1., 3., 3., 2., -1.])
        q = mx.array([3., 2., 1., 1., 0., -2.])
        ps = [p - mx.logsumexp(p)] * depth
        qs = [q - mx.logsumexp(q)] * depth
        drafts = [int(sampler(q[None]).item()) for _ in range(depth)]
        targets = [int(sampler(p[None]).item()) for _ in range(depth)]
        rows = {99: 'stale'} if reuse else None
        count = acceptance.accepted_count(drafts, targets, qs, ps,
            stochastic=True, sampler=sampler, filtered_rows=rows)
        correction = None
        if count < depth:
            fp, fq = rows[count] if reuse else (
                acceptance.accept_lp_for(sampler, ps[count]),
                acceptance.accept_lp_for(sampler, qs[count]))
            assert bool(mx.all(fp == acceptance.accept_lp_for(sampler, ps[count])).item())
            assert bool(mx.all(fq == acceptance.accept_lp_for(sampler, qs[count])).item())
            correction, _ = acceptance.residual_sample(fp, fq, sampler=sampler)
        if reuse:
            assert 99 not in rows and len(rows) <= depth <= 3
        outcomes.append((count, correction, sampler._vmlx_random_uniform()))
    assert outcomes[0] == outcomes[1]


def test_custom_sampler_never_publishes_reusable_rows(monkeypatch):
    sampler = make_sampler(temp=1., top_k=2)
    sampler._vmlx_acceptance_logprobs = lambda row: row
    sampler._vmlx_random_uniform = lambda: .9
    monkeypatch.setattr(acceptance, '_PACKED_ACCEPTANCE', True)
    rows = {99: 'stale'}
    acceptance.accepted_count([0], [0], [mx.log(mx.array([.8,.2]))],
        [mx.log(mx.array([.2,.8]))], stochastic=True, sampler=sampler,
        filtered_rows=rows)
    assert rows == {}


def test_mllm_rejection_reuses_exact_rows_without_filter_call(monkeypatch):
    from vmlx_engine import mllm_batch_generator as gen
    sampler = make_sampler(temp=1., top_k=2, seed=21)
    p, q = mx.log(mx.array([.2,.8])), mx.log(mx.array([.8,.2]))
    def forbidden(*args):
        raise AssertionError('filtered correction repeated acceptance filter')
    monkeypatch.setattr(acceptance, 'accept_lp_for', forbidden)
    monkeypatch.setattr(gen, '_NATIVE_MTP_STOCHASTIC_ACCEPT', True)
    token, token_id = gen._native_mtp_rejection_correction(
        mx.array([0]), 0, p, q, sampler, filtered_pair=(p,q))
    assert int(token.item()) == token_id == 1
    stats = gen.MLLMNativeMTPStats()
    stats.stochastic_reused_filter_pairs = 3
    assert stats.to_dict(request_id='probe', finish_reason='stop', final_depth=1)['stochastic_verify']['reused_filter_pairs'] == 3
