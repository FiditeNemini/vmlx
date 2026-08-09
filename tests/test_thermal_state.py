"""Thermal telemetry helper: perf log lines carry NSProcessInfo thermalState
so field slowdown reports can be split into throttle vs real regressions."""

import sys

from vmlx_engine.utils.thermal import _STATE_NAMES, thermal_state_name


def test_thermal_state_name_returns_known_value():
    name = thermal_state_name()
    assert name in set(_STATE_NAMES.values()) | {"unknown"}


def test_thermal_state_readable_on_darwin():
    if sys.platform != "darwin":
        return
    assert thermal_state_name() in set(_STATE_NAMES.values())


def test_thermal_state_name_is_cheap():
    import time

    thermal_state_name()
    t0 = time.perf_counter()
    for _ in range(100):
        thermal_state_name()
    assert (time.perf_counter() - t0) < 0.1
