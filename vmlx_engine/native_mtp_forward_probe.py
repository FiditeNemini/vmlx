"""Bounded, opt-in completed-forward attribution, not a throughput benchmark.

This probe serializes the first few head/verifier forwards. Its intervals
include host graph construction and GPU completion, not isolated kernel time.
Normal serving adds no evaluation or synchronization when the probe is off.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)
_LIMIT_PER_PHASE = 8
_PHASES = frozenset({"head", "verify"})
_AR_PHASES = frozenset({"ar_model", "ar_sample"})


class NativeMTPForwardProbe:
    def __init__(self, mx, metadata, input_ready_ms, started):
        self.mx = mx
        self.metadata = metadata
        self.input_ready_ms = input_ready_ms
        self.started = started
        self.finished = False

    def finish(self, *outputs):
        if self.finished:
            return
        # Do not retain model arrays in the request or the published record.
        arrays = tuple(x for x in outputs if x is not None)
        self.mx.eval(*arrays)
        self.mx.synchronize()
        completed = time.perf_counter()
        self.finished = True
        logger.info(
            ("MLLM AR completed forward %s" if self.metadata["phase"] in _AR_PHASES
             else "MLLM native MTP completed forward %s"),
            json.dumps({
                **self.metadata,
                "input_ready_ms": self.input_ready_ms,
                "forward_ready_ms": (completed - self.started) * 1000.0,
                "output_shapes": [list(x.shape) for x in arrays],
                "output_dtypes": [str(x.dtype) for x in arrays],
                "clock": "serialized_host_and_gpu_completion_wall",
                "perturbs_pipeline": True,
                "kernel_time": False,
            }, separators=(",", ":"), allow_nan=False),
        )


def start_native_mtp_forward_probe(request, phase, mx, *, inputs=(), **metadata):
    flag = ("VMLX_AR_FORWARD_PROBE" if phase in _AR_PHASES
            else "VMLX_NATIVE_MTP_FORWARD_PROBE")
    if os.environ.get(flag, "0") != "1":
        return None
    if phase not in _PHASES | _AR_PHASES:
        raise ValueError(f"unknown native MTP forward phase: {phase}")
    counts = getattr(request, "_native_mtp_forward_probe_counts", None)
    if counts is None:
        counts = {}
        request._native_mtp_forward_probe_counts = counts
    count = counts.get(phase, 0)
    if count >= _LIMIT_PER_PHASE:
        return None
    counts[phase] = count + 1
    before = time.perf_counter()
    # Materialize input dependencies before timing this forward. A preceding
    # async queue wait belongs to this separate interval, not to the kernel.
    arrays = tuple(x for x in inputs if x is not None)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()
    started = time.perf_counter()
    return NativeMTPForwardProbe(mx, {
        **metadata,
        "request_id": getattr(request, "request_id", None),
        "phase": phase,
        "sample": count + 1,
    }, (started - before) * 1000.0, started)
