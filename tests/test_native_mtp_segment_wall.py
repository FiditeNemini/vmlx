"""Segment ownership timing must exclude productive AR and count only popped tokens."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'vmlx_engine/mllm_batch_generator.py'


def load_function(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(node)
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


def state():
    return SimpleNamespace(
        stats=SimpleNamespace(span_finalized=False, span_seconds=0., init_emits=1,
                              draft_emits=2, bonus_emits=0, verify_emits=0),
        cycle_span_start=10., ladder_depth=2, depth_ceiling=3, depth=2,
        queue=[], mtp_cache=[], ar_fallback_reason='cost', ar_tier=None,
    )


@pytest.mark.parametrize('calibration', [False, True])
def test_actual_handoff_excludes_abandon_and_ar_work(calibration):
    clock = [10.5]
    mtp = state()
    published = []
    calls = []
    ns = {'time': SimpleNamespace(perf_counter=lambda: clock[0]),
          '_native_mtp_adaptive_policy': lambda: False}
    load_function('_native_mtp_finalize_span', ns)
    def expensive(*args):
        clock[0] += 100
        calls.append('abandon')
    def step(*args):
        clock[0] += 100
        calls.append('ar')
        return 'next', 'lp'
    req = SimpleNamespace(request_id='r', _native_mtp_state=mtp)
    class Tokens:
        def __getitem__(self, key):
            return self
    batch = SimpleNamespace(requests=[req], cache=[], y=Tokens(), logprobs=[])
    owner = SimpleNamespace(
        _abandon_pending_native_mtp_verify=expensive, _step=step,
        _stats=SimpleNamespace(record_native_mtp=lambda **kw: published.append(kw)),
        _observe_native_mtp_profile=lambda *args: calls.append('profile'))
    ns.update(self=owner, batch=batch, mtp_state=mtp, token=7,
              _native_mtp_ar_fallback_ready=lambda *args: (True, ''),
              _submit_decode_token_eval=lambda *args: None,
              _native_mtp_handoff_is_calibration=lambda s: calibration,
              _native_mtp_log_stats=lambda *args: None,
              _native_mtp_reentry_enabled=lambda: False,
              logger=SimpleNamespace(info=lambda *args: None))
    tree = ast.parse(SOURCE.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and 'ar_fallback_pending' in ast.unparse(n.test)
                  and any(isinstance(c, ast.Assign) and any(isinstance(t, ast.Name)
                          and t.id == '_handoff_t0' for t in c.targets) for c in n.body))
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(SOURCE), 'exec'), ns)
    assert calls[:2] == ['abandon', 'ar']
    assert mtp.stats.span_seconds == .5
    assert mtp.stats.span_finalized
    assert mtp.cycle_span_start == 10.
    assert mtp.stats.init_emits + mtp.stats.draft_emits == 3
    assert published[0]['finish_reason'] == ('ar_calibration' if calibration else 'fallback_to_ar')
    assert ('profile' in calls) is (not calibration)
    assert not hasattr(req, '_native_mtp_state')
    ns['_native_mtp_finalize_span'](mtp, now=999.)
    assert mtp.stats.span_seconds == .5


@pytest.mark.parametrize('anchor', [0, None, float('nan'), float('inf'), 11., 'bad'])
def test_invalid_anchor_does_not_invent_elapsed_time(anchor):
    ns = {'time': SimpleNamespace(perf_counter=lambda: 10.5),
          '_native_mtp_adaptive_policy': lambda: False}
    finalize = load_function('_native_mtp_finalize_span', ns)
    mtp = state()
    mtp.cycle_span_start = anchor
    mtp.queue = ['unemitted']
    finalize(mtp)
    assert mtp.stats.span_seconds == 0
    assert mtp.stats.span_finalized
    assert mtp.queue == ['unemitted']
    assert mtp.stats.draft_emits == 2


@pytest.mark.parametrize('rollback_fails', [False, True])
def test_terminal_wall_excludes_cleanup_and_failure_never_publishes(rollback_fails):
    clock = [10.5]
    mtp = state()
    mtp.pending_verify = object()
    mtp.terminal_snapshot = object()
    mtp.queue = ['not emitted']
    req = SimpleNamespace(_native_mtp_state=mtp)
    published, responses = [], []
    def rewind(*args):
        clock[0] += 100
        if rollback_fails:
            raise RuntimeError('rollback failed')
    ns = {'time': SimpleNamespace(perf_counter=lambda: clock[0]),
          '_native_mtp_adaptive_policy': lambda: False}
    load_function('_native_mtp_finalize_span', ns)
    ns.update(mtp_state_for_finish=mtp, req=req, batch=SimpleNamespace(cache=[]),
              self=SimpleNamespace(_rewind_native_mtp_terminal_boundary=rewind,
                  _stats=SimpleNamespace(record_native_mtp=lambda **kw: published.append(kw)),
                  _observe_native_mtp_profile=lambda *args: None),
              request_id='r', finish_reason='stop', uid=1, logprobs=[None], i=0,
              responses=responses, MLLMBatchResponse=lambda **kw: kw,
              logger=SimpleNamespace(error=lambda *args: None),
              _native_mtp_log_stats=lambda *args: None)
    tree = ast.parse(SOURCE.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'mtp_state_for_finish is not None')
    wrapper = ast.For(target=ast.Name(id='_once', ctx=ast.Store()),
                      iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                      body=[branch], orelse=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 str(SOURCE), 'exec'), ns)
    assert mtp.stats.span_seconds == .5
    assert not hasattr(req, '_native_mtp_state')
    if rollback_fails:
        assert not published
        assert responses[0]['finish_reason'] == 'error'
        assert responses[0]['prompt_cache'] is None
        assert mtp.queue == []
    else:
        assert len(published) == 1
        assert not responses
        assert mtp.queue == ['not emitted']


def test_health_serialization_reports_wall_separately_from_lazy_phase_sums():
    import dataclasses
    import typing
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'MLLMNativeMTPStats')
    ns = dict(vars(typing), dataclass=dataclasses.dataclass, field=dataclasses.field,
              _native_mtp_stats_depth_slots=lambda: 3,
              _native_mtp_trace_enabled=lambda: False,
              native_mtp_cache_lifecycle_snapshot=lambda **kw: kw)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    stats = ns['MLLMNativeMTPStats']()
    stats.cycles = 2
    stats.init_emits = 1
    stats.draft_emits = 2
    stats.verify_ms = 9000.
    stats.span_seconds = .5
    stats.span_finalized = True
    stats.stochastic_packed_probability_reads = 7
    result = stats.to_dict(request_id='r', finish_reason='fallback_to_ar', final_depth=1)
    assert result['confirmed_popped_tokens'] == 3
    assert result['confirmed_popped_tok_s'] == 6.
    assert result['span_seconds'] == .5
    assert result['stochastic_verify']['packed_probability_reads'] == 7
    assert result['timings_ms']['verify'] == 9000.
    stats.span_seconds = 0.
    assert stats.to_dict(request_id='r', finish_reason='error', final_depth=1)['confirmed_popped_tok_s'] is None


def test_cancel_freezes_before_discard_without_counting_queued_tokens():
    clock = [10.5]
    mtp = state()
    mtp.queue = [7, 8, 9]
    published = []
    ns = {'time': SimpleNamespace(perf_counter=lambda: clock[0]),
          '_native_mtp_adaptive_policy': lambda: False}
    load_function('_native_mtp_finalize_span', ns)
    def discard(*args):
        clock[0] += 100
        mtp.queue.clear()
    owner = SimpleNamespace(active_batch=SimpleNamespace(cache=[]),
            _abandon_pending_native_mtp_verify=discard,
            _stats=SimpleNamespace(record_native_mtp=lambda **kw: published.append(kw)))
    ns.update(self=owner, mtp_state=mtp, request=SimpleNamespace(request_id='r'),
              _native_mtp_log_stats=lambda *args: None,
              logger=SimpleNamespace(debug=lambda *args, **kw: None))
    tree = ast.parse(SOURCE.read_text())
    remove = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == 'remove' and 'cancelled' in ast.unparse(n))
    block = next(n for n in ast.walk(remove) if isinstance(n, ast.Try)
                 and n.body and isinstance(n.body[0], ast.Expr)
                 and isinstance(n.body[0].value, ast.Call)
                 and isinstance(n.body[0].value.func, ast.Name)
                 and n.body[0].value.func.id == '_native_mtp_finalize_span')
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), 'exec'), ns)
    assert mtp.stats.span_seconds == .5
    assert mtp.stats.init_emits + mtp.stats.draft_emits == 3
    assert mtp.queue == []
    assert published[0]['finish_reason'] == 'cancelled'


def test_decode_error_freezes_span_and_preserves_error_result():
    clock = [10.5]
    mtp = state()
    req = SimpleNamespace(request_id='r', _native_mtp_state=mtp)
    published = []
    owner = SimpleNamespace(active_batch=object(), language_model=None,
            _stats=SimpleNamespace(record_native_mtp=lambda **kw: published.append(kw)))
    ns = {'time': SimpleNamespace(perf_counter=lambda: clock[0]),
          '_native_mtp_adaptive_policy': lambda: False}
    load_function('_native_mtp_finalize_span', ns)
    def drop(*args):
        clock[0] += 100
    ns.update(self=owner, mtp_state=mtp, batch=SimpleNamespace(requests=[req],
              uids=[1], request_ids=['r']), exc=RuntimeError('original failure'),
              drop_parked_context=drop, _native_mtp_log_stats=lambda *args: None,
              logger=SimpleNamespace(error=lambda *args: None), prefill_errors=[],
              MLLMBatchResponse=lambda **kw: kw, mx=SimpleNamespace(zeros=lambda shape: None))
    tree = ast.parse(SOURCE.read_text())
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                   and 'MLLM native MTP decode failed for %s' in ast.unparse(n))
    body = [n for n in handler.body if not isinstance(n, ast.ImportFrom)]
    fn = ast.FunctionDef(name='run_error', args=ast.arguments(posonlyargs=[], args=[],
                         kwonlyargs=[], kw_defaults=[], defaults=[]),
                         body=body, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(SOURCE), 'exec'), ns)
    result = ns['run_error']()
    assert mtp.stats.span_seconds == .5
    assert published[0]['finish_reason'] == 'error'
    assert result[0]['finish_reason'] == 'error'
    assert 'original failure' in result[0]['error']
    assert owner.active_batch is None
    assert not hasattr(req, '_native_mtp_state')


def diagnostic_state():
    return SimpleNamespace(
        epoch=0, drafts=[1, 2],
        stats=SimpleNamespace(verify_ms=10., sample_ms=2., draft_ms=3.,
                              snapshot_ms=0., restore_ms=0., replay_ms=0.,
                              materialize_ms=1.),
    )


def diagnostic_helpers():
    ns = {}
    load_function('_native_mtp_timing_total_ms', ns)
    return ns, load_function('_native_mtp_diagnostic_cycle_wall', ns)


def test_cycle_diagnostic_accounts_fence_acceptance_and_signed_remainder():
    ns, observe = diagnostic_helpers()
    mtp = diagnostic_state()
    before = vars(mtp.stats).copy()
    assert observe(mtp, now=10., fence_ms=0., acceptance_ms=0.) is None
    mtp.stats.verify_ms += 20.
    row = observe(mtp, now=10.05, fence_ms=7., acceptance_ms=3.)
    assert row['observed_cycle_wall_ms'] == pytest.approx(50.)
    assert row['phase_scope_wall_ms'] == 20.
    assert row['unassigned_wall_ms'] == pytest.approx(20.)
    # Overlapping phase scopes must remain visible, not silently clamped.
    mtp.stats.verify_ms += 80.
    row = observe(mtp, now=10.06, fence_ms=2., acceptance_ms=1.)
    assert row['unassigned_wall_ms'] == pytest.approx(-73.)
    assert vars(mtp.stats) == {**before, 'verify_ms': 110.}
    assert ns['_native_mtp_timing_total_ms'](mtp.stats) == 116.


@pytest.mark.parametrize('transition', ['epoch', 'depth', 'reentry', 'clock', 'stats'])
def test_cycle_diagnostic_does_not_bridge_incompatible_intervals(transition):
    _, observe = diagnostic_helpers()
    mtp = diagnostic_state()
    assert observe(mtp, now=10., fence_ms=0., acceptance_ms=0.) is None
    now = 20.
    if transition == 'epoch':
        mtp.epoch += 1
    elif transition == 'depth':
        mtp.drafts.append(3)
    elif transition == 'reentry':
        # The real re-entry owner seeds a fresh state after the AR interval.
        mtp = diagnostic_state()
    elif transition == 'clock':
        now = 9.
    else:
        mtp.stats.verify_ms = 0.
    assert observe(mtp, now=now, fence_ms=1., acceptance_ms=1.) is None
    mtp.stats.verify_ms += 1.
    row = observe(mtp, now=now + .01, fence_ms=1., acceptance_ms=1.)
    assert row['observed_cycle_wall_ms'] == pytest.approx(10.)
    assert row['phase_scope_wall_ms'] == 1.
