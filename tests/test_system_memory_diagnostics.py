import logging
from types import SimpleNamespace

import pytest

from vmlx_engine.memory_status import log_system_memory


@pytest.mark.parametrize("available,percent", [(258, -87.9), (129, 10), (-1, 10), (64, float("nan"))])
def test_impossible_host_counters_are_unknown_not_capacity(caplog, available, percent):
    with caplog.at_level(logging.INFO):
        log_system_memory("before load", reader=lambda: SimpleNamespace(
            total=128, available=available, percent=percent,
        ))
    assert "availability unknown" in caplog.text
    assert "GB available" not in caplog.text
    assert "High system memory pressure" not in caplog.text


def test_valid_pressure_measurement_is_preserved(caplog):
    with caplog.at_level(logging.INFO):
        log_system_memory("after load", reader=lambda: SimpleNamespace(
            total=128 * 1024**3, available=4 * 1024**3, percent=96.9,
        ))
    assert "4.0GB available / 128.0GB total (96.9% used)" in caplog.text
    assert "High system memory pressure after load" in caplog.text


def test_reader_failure_is_diagnostic_only(caplog):
    def failed():
        raise OSError("unavailable")
    with caplog.at_level(logging.DEBUG):
        log_system_memory("before load", reader=failed)
    assert "unavailable" in caplog.text
