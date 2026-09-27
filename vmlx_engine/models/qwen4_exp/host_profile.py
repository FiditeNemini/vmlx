"""Bounded opt-in host attribution; never inserts a GPU evaluation fence."""

import cProfile
import functools
import io
import logging
import os
import pstats
import threading

logger = logging.getLogger(__name__)


def profile_decode_forward(function):
    """Profile 32 calls of a selected small row count per thread when enabled.

    With the flag absent, return the original function: no wrapper or hot-path
    environment lookup. Timings include Python/native host calls and any waits
    those calls already perform; they are not GPU kernel timings or throughput.
    """
    if os.environ.get("VMLX_QWEN4_HOST_PROFILE") != "1":
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
        if count == 0:
            local.profiler = cProfile.Profile()
        local.count = count + 1
        local.profiler.enable()
        try:
            return function(self, inputs, *args, **kwargs)
        finally:
            local.profiler.disable()
            if local.count == 32:
                report = io.StringIO()
                stats = pstats.Stats(local.profiler, stream=report).strip_dirs()
                stats.sort_stats("tottime").print_stats(40)
                stats.sort_stats("cumulative").print_stats(30)
                logger.info(
                    "QWEN4_HOST_PROFILE calls=32 scope=forward_host_only "
                    "instrumented=true gpu_fences_added=false rows=%s\n%s",
                    rows,
                    report.getvalue(),
                )
                del local.profiler

    return wrapped
