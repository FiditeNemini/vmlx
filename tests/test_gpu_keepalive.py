"""GPU keep-alive while a large model is resident (2027cf82, 75149a79).

Measured 2026-10-09 (Naive B97, 95.9 GiB resident): after >= 1.5 s of GPU idle the next call paid a 0.65-1.25 s wake
penalty; a tiny GPU op every <= 1 s removed it. The beat must run while requests arrive, stop by itself after the idle
timeout (no power drain on an abandoned server), restart on the next touch, honour VMLX_GPU_KEEPALIVE=0, and be
released on model unload.
"""
import time

import pytest

pytest.importorskip("mlx.core")

from vmlx_engine.utils import gpu_keepalive as KA  # noqa: E402


def _wait(pred, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_beats_while_touched_and_stops_after_idle_timeout_then_restarts():
    ka = KA.KeepAlive(period_s=0.02, idle_timeout_s=0.25).start()
    try:
        assert _wait(lambda: ka.beats >= 3)
        assert _wait(lambda: not ka._th.is_alive(), timeout=2.0)      # idle timeout: the thread exits by itself
        stopped_at = ka.beats
        time.sleep(0.1)
        assert ka.beats == stopped_at                                  # no beats after the idle stop
        ka.touch()                                                     # next request restarts it
        assert _wait(lambda: ka.beats > stopped_at)
    finally:
        ka.stop()
    assert not ka._th.is_alive()


def test_global_honours_disable_env_and_is_released_on_unload(monkeypatch):
    monkeypatch.setattr(KA, "_GLOBAL", None)
    monkeypatch.setenv("VMLX_GPU_KEEPALIVE", "0")
    KA.touch_global()
    assert KA._GLOBAL is None                                          # disabled: nothing started
    monkeypatch.setenv("VMLX_GPU_KEEPALIVE", "1")
    monkeypatch.setenv("VMLX_GPU_KEEPALIVE_IDLE_S", "5")
    KA.touch_global()
    instance = KA._GLOBAL
    assert instance is not None and instance.idle_timeout_s == 5.0
    KA.touch_global()
    assert KA._GLOBAL is instance                                      # one process-wide heartbeat
    KA.stop_global()
    assert KA._GLOBAL is None and not instance._th.is_alive()          # unload releases it


def test_engines_touch_per_request_and_release_on_unload():
    import inspect

    from vmlx_engine.engine import batched, simple

    for module in (batched, simple):
        src = inspect.getsource(module)
        assert "_gpu_keepalive_touch()" in src and "stop_global()" in src, module.__name__
