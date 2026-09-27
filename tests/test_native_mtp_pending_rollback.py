# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow tests for pending native MTP rollback; no MLX imports."""

import ast
from collections import deque
import logging
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest


def repository_root():
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        if (root / "vmlx_engine/mllm_batch_generator.py").is_file():
            return root
    raise AssertionError("Run from the repository or place this file in tests/")


ROOT = repository_root()
SOURCE = ROOT / "vmlx_engine/mllm_batch_generator.py"
TREE = ast.parse(SOURCE.read_text())
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and any(
    isinstance(child, ast.FunctionDef)
    and child.name == "_abandon_pending_native_mtp_verify" for child in node.body
))
SCHEDULER = ast.parse((ROOT / "vmlx_engine/mllm_scheduler.py").read_text())


def execute(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)


@pytest.fixture
def probe(monkeypatch):
    trace = []
    namespace = {
        "logger": logging.getLogger(__name__), "__package__": "vmlx_engine",
        "time": time,
        "_native_mtp_finalize_span": lambda *args, **kwargs: trace.append("finalize_span"),
    }
    priming = ModuleType("vmlx_engine.native_mtp_prompt_priming")
    priming.drop_parked_context = lambda *args: trace.append("drop_parked")
    monkeypatch.setitem(sys.modules, priming.__name__, priming)
    names = {"_abandon_pending_native_mtp_verify", "_rewind_native_mtp_terminal_boundary", "remove"}
    methods = [node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name in names]
    execute(methods, namespace)
    owner = SimpleNamespace(language_model=None, unprocessed_requests=[])
    for method in methods:
        setattr(owner, method.name, namespace[method.name].__get__(owner))
    state = SimpleNamespace(
        pending_verify={"snapshot": [None], "n_inputs": 2},
        terminal_snapshot=object(), queue=deque(), next_main=7,
    )
    return owner, state, namespace, trace


def install_restore(namespace, trace, mode):
    def restore(*args):
        trace.append("restore")
        if mode == "exception":
            raise RuntimeError("injected restore failure")
        return mode == "success"
    namespace["_native_mtp_restore_replay_cache"] = restore


@pytest.mark.parametrize("mode", ["false", "exception", "missing_cache"])
def test_abandon_refuses_failed_restore_and_retains_pending(probe, mode):
    owner, state, namespace, trace = probe
    install_restore(namespace, trace, mode)
    pending = state.pending_verify
    with pytest.raises(RuntimeError, match="rollback"):
        owner._abandon_pending_native_mtp_verify(
            state, None if mode == "missing_cache" else [object()]
        )
    assert state.pending_verify is pending


def test_success_clears_pending_and_no_pending_is_noop(probe):
    owner, state, namespace, trace = probe
    install_restore(namespace, trace, "success")
    owner._abandon_pending_native_mtp_verify(state, [object()])
    assert state.pending_verify is None
    owner._abandon_pending_native_mtp_verify(state, [object()])
    assert trace == ["restore"]


@pytest.mark.parametrize("mode", ["false", "exception"])
def test_ar_handoff_does_not_step_after_failed_restore(probe, mode):
    owner, state, namespace, trace = probe
    install_restore(namespace, trace, mode)
    ready = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_native_mtp_ar_fallback_ready")
    namespace["_native_mtp_scalar_id"] = lambda value: value
    execute([ready], namespace)
    handoff = next(node for node in ast.walk(CLASS) if isinstance(node, ast.If) and any(
        isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Attribute)
        and item.value.func.attr == "_abandon_pending_native_mtp_verify"
        for item in node.body
    ))
    class Input:
        def __getitem__(self, key):
            return self
    def step(*args):
        trace.append("AR step")
        return Input(), []
    owner._step = step
    namespace.update(self=owner, mtp_state=state, token=7,
                     batch=SimpleNamespace(cache=[SimpleNamespace()], y=Input()))
    with pytest.raises(RuntimeError, match="rollback"):
        # Keep the real prefix through the first AR step; telemetry additions
        # must not shift a positional slice past rollback or omit the step.
        step_index = next(i for i, statement in enumerate(handoff.body) if any(
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "self._step"
            for call in ast.walk(statement)
        ))
        execute(handoff.body[:step_index + 1], namespace)
    assert "AR step" not in trace


@pytest.mark.parametrize("mode", ["false", "exception", "missing_cache"])
def test_terminal_error_skips_success_capture_and_removes_batch(probe, mode):
    owner, state, namespace, trace = probe
    install_restore(namespace, trace, mode)
    class NoSuccessStats:
        def __getattr__(self, name):
            raise AssertionError("Failed rollback reached terminal success statistics")

        def __setattr__(self, name, value):
            raise AssertionError("Failed rollback reached terminal success statistics")
    state.ladder_depth = 1
    state.stats = NoSuccessStats()
    owner._stats = NoSuccessStats()
    request = SimpleNamespace(
        request_id="request", _native_mtp_state=state,
        _media_clean_prefix_cache=object(), _mixed_swa_boundary=object(),
    )
    batch = SimpleNamespace(cache=None if mode == "missing_cache" else [object()])
    owner.active_batch = batch
    next_method = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_next")
    terminal = next(node for node in ast.walk(next_method) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "finish_reason is not None" and any(
                        isinstance(item, ast.ImportFrom) and item.module == "native_mtp_prompt_priming"
                        for item in node.body))
    cleanup = next(node for node in next_method.body if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "end_idx")
    namespace.update(
        self=owner, req=request, batch=batch, request_id="request", uid=3, i=0,
        logprobs=[object()], finish_reason="stop", responses=[], end_idx=[0], keep_idx=[],
        MLLMBatchResponse=lambda **kwargs: SimpleNamespace(**kwargs),
        mx=SimpleNamespace(clear_cache=lambda: trace.append("allocator cleanup")),
    )
    wrapper = ast.For(
        target=ast.Name(id="once", ctx=ast.Store()),
        iter=ast.List(elts=[ast.Constant(value=1)], ctx=ast.Load()),
        body=[terminal], orelse=[],
    )
    execute([wrapper, cleanup], namespace)
    assert owner.active_batch is None
    assert len(namespace["responses"]) == 1
    response = namespace["responses"][0]
    assert response.finish_reason == "error"
    assert response.prompt_cache is None
    assert response.error.startswith("NativeMTPError:")
    assert not hasattr(request, "_native_mtp_state")
    assert not hasattr(request, "_media_clean_prefix_cache")

    gate = next(node for node in ast.walk(SCHEDULER) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "not is_error" and any(
                    ast.unparse(item) == "request.output_tokens.append(response.token)"
                    for item in node.body))
    scheduler_request = SimpleNamespace(output_tokens=[11, 12])
    execute([gate], {"is_error": True, "request": scheduler_request, "response": response})
    assert scheduler_request.output_tokens == [11, 12]

    # Execute the actual cache extraction gate; no terminal payload may enter
    # scheduler storage. This is not a full disk-store integration test.
    cache_gate = next(node for node in ast.walk(SCHEDULER) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "getattr(response, 'prompt_cache', None) is not None")
    execute([cache_gate], {
        "request": scheduler_request, "response": response,
        "request_id": "request", "finish_reason": "error", "logger": logging.getLogger(__name__),
    })
    assert not hasattr(scheduler_request, "_extracted_cache")


def test_cancellation_discards_request_even_if_rollback_fails(probe):
    owner, state, namespace, trace = probe
    install_restore(namespace, trace, "false")
    request = SimpleNamespace(uid=1, request_id="request", _native_mtp_state=state)
    owner.active_batch = SimpleNamespace(uids=[1], requests=[request], cache=[object()])
    owner.remove([1])
    assert owner.active_batch is None
    assert not hasattr(request, "_native_mtp_state")
    assert trace[0] == "drop_parked"
