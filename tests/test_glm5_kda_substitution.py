# SPDX-License-Identifier: Apache-2.0
"""Bounded FP32 solver equivalence, fallback safety and startup identity."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
mx = pytest.importorskip("mlx.core")
from vmlx_engine.metal import glm5_kda_substitution as fused
from vmlx_engine.models.glm5_next import kda
from vmlx_engine import glm5_prefill_policy as policy


def exact(a, b):
    mx.eval(a, b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)))


@pytest.fixture
def qualified(monkeypatch):
    if not fused._compatible_runtime():
        pytest.skip("MLX 0.32.2/0.32.3 on M5 Max numerical qualification")
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)
    monkeypatch.setattr(fused, "_CALLS", 0)


def stock(a):
    a = mx.array(a)
    for i in range(1, 64):
        upd = mx.sum(a[..., i, :, None] * a[..., :, :i], axis=-2)
        a[..., i, :i] = a[..., i, :i] + upd
    return a


@pytest.mark.parametrize("case", ["random", "signed_zero", "cancellation", "tiny"])
def test_triangular_bits_and_input_immutability(qualified, case):
    mx.random.seed(51)
    a = mx.tril(mx.random.normal((1, 64, 2, 64, 64)) * .025, k=-1)
    if case == "signed_zero":
        a = a * mx.array(-0.0, mx.float32)
    elif case == "cancellation":
        a = mx.tril(mx.where(mx.arange(64) % 2, .125, -.125) * mx.ones_like(a), k=-1)
    elif case == "tiny":
        a = a * 1e-25
    original = mx.array(a)
    expected = stock(a)
    actual = fused.kda_substitution(a, enabled=True)
    assert actual is not None and fused._OBSERVED
    exact(expected, actual)
    exact(a, original)


@pytest.mark.parametrize("tokens", [817, 1024, 2048])
def test_full_kda_state_and_decode(qualified, monkeypatch, tokens):
    mx.random.seed(tokens)
    shape = (1, tokens, 64, 128)
    q, k = [kda.l2norm(mx.random.normal(shape).astype(mx.bfloat16)) for _ in range(2)]
    v = mx.random.normal(shape).astype(mx.bfloat16)
    g = -5 * mx.sigmoid(mx.random.normal(shape))
    beta = mx.sigmoid(mx.random.normal(shape[:-1]))
    initial = mx.random.normal((1, 64, 128, 128)) * .01
    monkeypatch.setattr(policy, "_KDA_SUBSTITUTION_REQUESTED", False)
    oe, se = kda.kda_chunked(q, k, v, g, beta, initial)
    mx.eval(oe, se)
    monkeypatch.setattr(policy, "_KDA_SUBSTITUTION_REQUESTED", True)
    oa, sa = kda.kda_chunked(q, k, v, g, beta, initial)
    exact(oe, oa); exact(se, sa)
    assert fused._OBSERVED
    x = mx.random.normal((1, 64, 128))
    args = (kda.l2norm(x), kda.l2norm(x), x, -mx.ones_like(x), mx.full((1, 64), .5))
    oe, se = kda.kda_step(*args, se)
    oa, sa = kda.kda_step(*args, sa)
    exact(oe, oa); exact(se, sa)


def test_ineligible_and_failed_launch_keep_original(qualified, monkeypatch):
    a = mx.zeros((1, 64, 1, 64, 64))
    def broken():
        raise RuntimeError("controlled compile failure")
    monkeypatch.setattr(fused, "_kernel", broken)
    assert fused.kda_substitution(a, enabled=False) is None
    for bad in [a[0], a[:, :63], a[..., :63], a.astype(mx.bfloat16), mx.zeros((1,64,33,64,64)), mx.zeros((2,64,1,64,64))]:
        assert fused.kda_substitution(bad, enabled=True) is None
        assert not fused._FAILED
    original = mx.array(a)
    assert fused.kda_substitution(a, enabled=True) is None
    assert fused._FAILED
    exact(a, original)


def test_runtime_fallback_before_kernel(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    monkeypatch.setattr(fused, "_kernel", lambda: pytest.fail("unqualified runtime launched"))
    assert fused.kda_substitution(mx.zeros((1,64,1,64,64)), enabled=True) is None


@pytest.mark.parametrize("value,expected", [(None,True),("0",False),("1",True),("true",False),("2",False)])
def test_startup_policy_and_health_freeze(value, expected):
    env=dict(os.environ)
    env.pop("VMLX_GLM5_KDA_SUBSTITUTION", None)
    if value is not None: env["VMLX_GLM5_KDA_SUBSTITUTION"]=value
    code='''import os,json
from vmlx_engine.glm5_prefill_policy import glm5_kda_substitution_requested as requested
from vmlx_engine.acceleration_contract import build_acceleration_contract
before=requested()
os.environ['VMLX_GLM5_KDA_SUBSTITUTION']='0' if before else '1'
row=next(r for r in build_acceleration_contract('glm5_next')['features'] if r['id']=='kda_substitution')
print(json.dumps([before,requested(),row['requested']]))'''
    result=json.loads(subprocess.check_output([sys.executable,"-c",code],env=env,text=True))
    assert result == [expected]*3


def test_glm_only_namespace_and_observed_health(qualified, monkeypatch):
    from vmlx_engine.prefix_cache import compute_model_cache_key
    from vmlx_engine.acceleration_contract import build_acceleration_contract
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp"):
        model=SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.setattr(policy,"_KDA_SUBSTITUTION_REQUESTED",False)
        off=compute_model_cache_key(model)
        monkeypatch.setattr(policy,"_KDA_SUBSTITUTION_REQUESTED",True)
        on=compute_model_cache_key(model)
        assert (on != off) == family.startswith("glm5_next")
    result=fused.kda_substitution(mx.zeros((1,64,1,64,64)))
    mx.eval(result)
    status=fused.kda_substitution_status()
    row=next(r for r in build_acceleration_contract('glm5_next',{'features':{'kda_substitution':status}})['features'] if r['id']=='kda_substitution')
    assert row['state']=='active_observed'
    assert status['observed_calls']==1 and status['enqueued_calls']==1


def test_legacy_runtime_fingerprint_isolates_opt_in(monkeypatch):
    from vmlx_engine.prefix_cache import _resolve_runtime_cache_fingerprint
    monkeypatch.setattr(policy, "_KDA_SUBSTITUTION_REQUESTED", False)
    off = _resolve_runtime_cache_fingerprint()
    monkeypatch.setattr(policy, "_KDA_SUBSTITUTION_REQUESTED", True)
    on = _resolve_runtime_cache_fingerprint()
    assert off != on
    monkeypatch.setenv("VMLX_GLM5_KDA_SUBSTITUTION", "0")
    assert _resolve_runtime_cache_fingerprint() == on
