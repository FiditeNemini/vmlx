"""Bounded opt-in host attribution; never inserts a GPU evaluation fence."""

import functools
import io
import json
import logging
import os
import pstats
import profile
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Captured at import, like the existing diagnostic. Normal model execution
# keeps the direct async_eval call and never reads environment variables.
SUBMISSION_PROFILE_ENABLED = (
    os.environ.get("VMLX_QWEN4_HOST_PROFILE") == "1"
    and os.environ.get("VMLX_QWEN4_HOST_PROFILE_MODE") == "phases"
    and os.environ.get("VMLX_QWEN4_HOST_PROFILE_ROWS", "1") in {"1", "2", "3", "4"}
)
_phase_local = threading.local()


def profile_submission(submit, value):
    """Time an existing submission on the selected calling thread, without fences."""
    active = getattr(_phase_local, "active", None)
    if active is None:
        return submit(value)
    wall = time.perf_counter()
    cpu = time.thread_time()
    try:
        return submit(value)
    finally:
        active["submission_cpu_s"] += time.thread_time() - cpu
        active["submission_wall_s"] += time.perf_counter() - wall
        active["submissions"] += 1


def _phase_forward(local, count, rows, function, *args, **kwargs):
    totals = getattr(local, "totals", None)
    if totals is None:
        totals = local.totals = dict(
            forward_wall_s=0.0, forward_cpu_s=0.0,
            submission_wall_s=0.0, submission_cpu_s=0.0, submissions=0,
            failed_calls=0,
        )
    previous = getattr(_phase_local, "active", None)
    _phase_local.active = totals
    wall = time.perf_counter()
    cpu = time.thread_time()
    local.count = count + 1
    try:
        return function(*args, **kwargs)
    except BaseException:
        totals["failed_calls"] += 1
        raise
    finally:
        totals["forward_cpu_s"] += time.thread_time() - cpu
        totals["forward_wall_s"] += time.perf_counter() - wall
        _phase_local.active = previous
        if local.count == 32:
            report = dict(totals)
            for clock in ("wall", "cpu"):
                report[f"other_host_{clock}_s"] = (
                    totals[f"forward_{clock}_s"] - totals[f"submission_{clock}_s"]
                )
            logger.info(
                "QWEN4_HOST_PHASES calls=32 rows=%s instrumented=true "
                "gpu_fences_added=false timing=host_only data=%s",
                rows, json.dumps(report, sort_keys=True),
            )
            del local.totals


def profile_decode_forward(function):
    """Profile 32 calls of a selected small row count per thread when enabled.

    With the flag absent, return the original function: no wrapper or hot-path
    environment lookup. Timings include Python/native host calls and any waits
    those calls already perform; they are not GPU kernel timings or throughput.
    """
    if os.environ.get("VMLX_QWEN4_HOST_PROFILE") != "1":
        return function
    mode = os.environ.get("VMLX_QWEN4_HOST_PROFILE_MODE", "python")
    if mode not in {"python", "phases"}:
        logger.warning("Qwen host profile disabled: mode must be python or phases")
        return function
    # One row retains the existing decode diagnostic. Native MTP verification
    # uses depth + 1 rows; select it without profiling an entire prefill.
    rows = os.environ.get("VMLX_QWEN4_HOST_PROFILE_ROWS", "1")
    if rows not in {"1", "2", "3", "4"}:
        logger.warning("Qwen host profile disabled: rows must be 1, 2, 3 or 4")
        return function
    selected_shape = (1, int(rows))
    local = threading.local()

    @functools.wraps(function)
    def wrapped(self, inputs, *args, **kwargs):
        count = getattr(local, "count", 0)
        if count >= 32 or tuple(getattr(inputs, "shape", ())) != selected_shape:
            return function(self, inputs, *args, **kwargs)
        if selected_shape[1] > 1 and kwargs.get("return_hidden") is not True:
            # The native verifier explicitly requests hidden state. A short
            # ordinary prefill of the same shape must not consume its budget.
            return function(self, inputs, *args, **kwargs)
        if sys.getprofile() is not None:
            return function(self, inputs, *args, **kwargs)
        if getattr(_phase_local, "active", None) is not None:
            # Do not double-count a recursively entered forward call.
            return function(self, inputs, *args, **kwargs)
        if mode == "phases":
            return _phase_forward(local, count, rows, function, self, inputs,
                                  *args, **kwargs)
        # cProfile on the bundled Python can mix worker-thread events into
        # one stack. Use the calling-thread hook, with a fresh stack per call;
        # reusing profile.Profile.runcall across calls also corrupts its stack.
        profiler = profile.Profile(timer=time.perf_counter)
        local.count = count + 1
        try:
            return profiler.runcall(function, self, inputs, *args, **kwargs)
        finally:
            current = pstats.Stats(profiler, stream=io.StringIO())
            if count == 0:
                local.stats = current
            else:
                local.stats.add(current)
            if local.count == 32:
                report = io.StringIO()
                stats = local.stats
                stats.stream = report
                stats.strip_dirs()
                stats.sort_stats("tottime").print_stats(40)
                stats.sort_stats("cumulative").print_stats(30)
                logger.info(
                    "QWEN4_HOST_PROFILE calls=32 scope=forward_host_only "
                    "instrumented=true gpu_fences_added=false rows=%s "
                    "backend=thread_profile clock=wall\n%s",
                    rows,
                    report.getvalue(),
                )
                del local.stats

    return wrapped
