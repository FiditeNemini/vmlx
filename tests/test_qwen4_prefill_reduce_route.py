"""CPU source checks for candidate reachability; no numeric qualification claim."""

import os
from types import SimpleNamespace

import pytest

from .test_cache_cleanup_phase_timing import ROOT, load_functions


def fixture(monkeypatch, enabled=True):
    monkeypatch.setenv("VMLX_QWEN4_PREFILL_REDUCE", "1" if enabled else "0")
    events = []

    class Projection:
        def __init__(self, name, k, n):
            self.name, self.input_dims, self.output_dims = name, k, n
            self.mode, self.bits, self.group_size = "affine", 4, 64
            self.weight = SimpleNamespace(dtype="u32", ndim=3, shape=(512, n, k // 8))
            self.scales = self.biases = SimpleNamespace(dtype="f16")

        def __contains__(self, key):
            return False

        def __call__(self, x, ids, sorted_indices=False):
            events.append((self.name, x, ids, sorted_indices))
            return self.name

    class Switch:
        training = False

        def __init__(self):
            self.up_proj = Projection("up", 2560, 640)
            self.gate_proj = Projection("gate", 2560, 640)
            self.down_proj = Projection("down", 640, 2560)

        def activation(self, up, gate):
            events.append(("activation", up, gate))
            return "activated"

    def expand(x, axes):
        events.append(("expand", axes))
        return "expanded"

    def sort(x, ids):
        events.append(("sort", x))
        return "sorted_x", "sorted_ids", "inverse"

    def reduce(y, inv, scores):
        events.append(("reduce", y, inv))
        return "weighted"

    ns = dict(
        os=os,
        SwitchGLU=Switch,
        QuantizedSwitchLinear=Projection,
        mx=SimpleNamespace(
            float16="f16", uint32="u32", int32="i32", expand_dims=expand
        ),
        _available=lambda: True,
        _gather_sort=sort,
        weighted_unsort=reduce,
        _record_submission=lambda *args: None,
    )
    load_functions(
        ROOT / "metal/qwen4_prefill_reduce.py", {"requested", "prefill_reduce"}, ns
    )
    x = SimpleNamespace(ndim=3, shape=(1, 33, 2560), size=33 * 2560, dtype="f16")
    ids = SimpleNamespace(shape=(1, 33, 10), dtype="u32")
    scores = SimpleNamespace(shape=ids.shape, dtype="f16")
    return ns, Switch(), x, ids, scores, events


def test_native_projections_preserve_sort_activation_and_quant_parameters(monkeypatch):
    ns, switch, x, ids, scores, events = fixture(monkeypatch)
    # The epilogue never decodes weight bits or picks another projection kernel.
    for bits, group in [(2, 64), (3, 128), (4, 64), (5, 64), (6, 128), (8, 64)]:
        for p in (switch.up_proj, switch.gate_proj, switch.down_proj):
            p.bits, p.group_size = bits, group
        assert ns["prefill_reduce"](switch, x, ids, scores) == "weighted"
        assert events[-4:] == [
            ("gate", "sorted_x", "sorted_ids", True),
            ("activation", "up", "gate"),
            ("down", "activated", "sorted_ids", True),
            ("reduce", "down", "inverse"),
        ]
        assert all(
            (p.bits, p.group_size) == (bits, group)
            for p in (switch.up_proj, switch.gate_proj, switch.down_proj)
        )


@pytest.mark.parametrize(
    "reason",
    [
        "off",
        "hardware",
        "empty",
        "decode",
        "bf16",
        "scores_fp32",
        "custom",
        "training",
        "nonaffine",
        "projection_fp32",
        "indices",
    ],
)
def test_unsupported_route_falls_back_before_any_tensor_operation(monkeypatch, reason):
    ns, switch, x, ids, scores, events = fixture(monkeypatch, reason != "off")
    if reason == "hardware":
        ns["_available"] = lambda: False
    elif reason == "empty":
        x.size = 0
    elif reason == "decode":
        x.size = 2560
    elif reason == "bf16":
        x.dtype = "bf16"
    elif reason == "scores_fp32":
        scores.dtype = "fp32"
    elif reason == "custom":
        switch = SimpleNamespace(**vars(switch))
    elif reason == "training":
        switch.training = True
    elif reason == "nonaffine":
        switch.up_proj.mode = "mxfp4"
    elif reason == "projection_fp32":
        switch.down_proj.scales = SimpleNamespace(dtype="fp32")
    elif reason == "indices":
        ids.shape = (1, 33, 8)
    assert ns["prefill_reduce"](switch, x, ids, scores) is None
    assert events == []


def test_default_affine_owner_reaches_candidate_after_existing_priority_routes(
    monkeypatch,
):
    ns, switch, x, ids, scores, events = fixture(monkeypatch)
    ns.update(
        aligned_switchglu=lambda *a: None,
        scatter_route_switchglu=lambda *a: None,
        affine_moe_pair_activation=lambda *a: (None, False),
        _OK_ATTR="full",
        _EXACT_OK_ATTR="exact",
        _D=2560,
        _K=10,
    )
    load_functions(
        ROOT / "metal/qwen4_affine_moe_decode.py", {"qwen4_affine_switchglu"}, ns
    )
    assert ns["qwen4_affine_switchglu"](switch, x, ids, scores) == ("weighted", True)
    assert events[-1] == ("reduce", "down", "inverse")
