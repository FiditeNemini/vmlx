"""Selected-page advice admission and accounting; no MLX/model imports."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# The package initializer imports MLX/model code; this stdlib-only helper can
# be tested without initializing that unrelated runtime.
_spec = importlib.util.spec_from_file_location(
    "qwen4_page_read_advice_under_test",
    Path(__file__).parents[1] / "vmlx_engine/models/qwen4_exp/page_read_advice.py",
)
advice = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(advice)


class Backend:
    def __init__(self, resident=False, fail_probe=None, fail_hint=None):
        self.is_resident = resident
        self.fail_probe = fail_probe
        self.fail_hint = fail_hint
        self.probes = []
        self.hints = []
        self.events = []

    def resident(self, address, size):
        self.probes.append((address, size))
        self.events.append("probe")
        if len(self.probes) == self.fail_probe:
            raise OSError(12, "test probe failure")
        return self.is_resident

    def hint(self, fd, offset, count):
        self.events.append("hint")
        if len(self.hints) + 1 == self.fail_hint:
            raise OSError(5, "test hint failure")
        self.hints.append((fd, offset, count))


@pytest.fixture
def readers(tmp_path):
    owned = []

    def make(*, offset=19, rows=8, width=32, name=None):
        path = tmp_path / (name or f"table-{len(owned)}")
        path.write_bytes(bytes((i % 251 for i in range(offset + rows * width))))
        mm = np.memmap(path, mode="r", dtype=np.uint8, offset=offset, shape=(rows, width))
        fd = os.open(path, os.O_RDONLY)
        reader = SimpleNamespace(
            mm=mm, shape=mm.shape, row_bytes=width, data_offset=offset,
            _pread_file=SimpleNamespace(_fileno=lambda: fd),
        )
        owned.append((mm, fd))
        return reader

    yield make
    for mm, fd in owned:
        mm._mmap.close()
        os.close(fd)


def advisor(backend=None):
    return advice.SelectedPageReadAdvisor(os.sysconf("SC_PAGESIZE"), backend or Backend())


def test_cold_deduplicated_pages_alignment_and_eof(readers):
    reader = readers()
    backend = Backend()
    result = advisor(backend).advise([(reader, [0, 0, 1, 7])])
    assert result == dict(rows=3, pages=1, resident=0, hinted=1,
                          bytes=19 + 8 * 32, status="advised")
    assert backend.probes[0][0] % os.sysconf("SC_PAGESIZE") == 0
    assert backend.hints[0][1:] == (0, 19 + 8 * 32)
    assert backend.events == ["probe", "hint"]


def test_resident_pages_get_no_hint_and_reader_bytes_unchanged(readers):
    reader = readers()
    before = np.asarray(reader.mm).copy()
    backend = Backend(resident=True)
    result = advisor(backend).advise([(reader, [1, 2])])
    assert result["status"] == "resident"
    assert result["resident"] == result["pages"] == 1
    assert result["hinted"] == result["bytes"] == 0
    assert not backend.hints
    np.testing.assert_array_equal(reader.mm, before)


def test_multi_file_and_same_file_pages_deduplicate_before_hints(readers):
    page = os.sysconf("SC_PAGESIZE")
    first = readers(offset=page - 8, rows=2, width=16)
    second = readers(offset=3, rows=1, width=8)
    backend = Backend()
    result = advisor(backend).advise([(first, [0, 1]), (first, [0]), (second, [0])])
    assert result["pages"] == result["hinted"] == 3
    assert backend.events == ["probe"] * 3 + ["hint"] * 3
    assert [hint[1:] for hint in backend.hints] == [(0, page), (page, 24), (0, 11)]


@pytest.mark.parametrize("change", [
    lambda r: setattr(r, "row_bytes", r.row_bytes + 1),
    lambda r: setattr(r, "data_offset", r.data_offset + 1),
    lambda r: setattr(r, "shape", (1,)),
    lambda r: setattr(r, "mm", r.mm[:, ::2]),
    lambda r: setattr(r, "_pread_file", None),
])
def test_bad_layout_never_probes_or_hints(readers, change):
    reader = readers()
    change(reader)
    backend = Backend()
    result = advisor(backend).advise([(reader, [0])])
    assert result["status"] in {"unsupported", "error"}
    assert not backend.events


@pytest.mark.parametrize("indices", [[-1], [8], [0.5], [0, 8]])
def test_bad_rows_fail_open_before_any_hint(readers, indices):
    backend = Backend()
    result = advisor(backend).advise([(readers(), indices)])
    assert result["status"] in {"unsupported", "error"}
    assert not backend.events


@pytest.mark.parametrize("cap", ["pages", "bytes"])
def test_caps_apply_to_all_selected_pages_even_if_resident(readers, monkeypatch, cap):
    page = os.sysconf("SC_PAGESIZE")
    reader = readers(offset=0, rows=3, width=page)
    if cap == "pages":
        monkeypatch.setattr(advice, "_MAX_PAGES", 2)
    else:
        monkeypatch.setattr(advice, "_MAX_BYTES", 2 * page)
    backend = Backend(resident=True)
    result = advisor(backend).advise([(reader, [0, 1, 2])])
    assert result["status"] == "limit"
    assert result["pages"] == 2
    assert not backend.events


def test_mincore_failure_issues_no_hints(readers):
    page = os.sysconf("SC_PAGESIZE")
    backend = Backend(fail_probe=2)
    result = advisor(backend).advise([(readers(offset=0, width=page, rows=2), [0, 1])])
    assert result["status"] == "error"
    assert result["hinted"] == result["bytes"] == 0
    assert result["error"] == "OSError:errno=12"
    assert not backend.hints


@pytest.mark.parametrize("failure,expected", [(1, "error"), (2, "partial")])
def test_advice_failure_reports_only_successful_submissions(readers, failure, expected):
    page = os.sysconf("SC_PAGESIZE")
    backend = Backend(fail_hint=failure)
    result = advisor(backend).advise([(readers(offset=0, width=page, rows=2), [0, 1])])
    assert result["status"] == expected
    assert result["hinted"] == failure - 1
    assert result["bytes"] == (failure - 1) * page
    assert result["error"] == "OSError:errno=5"


def test_wrong_descriptor_and_truncated_file_rejected(readers):
    first, other = readers(), readers()
    backend = Backend()
    first._pread_file = other._pread_file
    assert advisor(backend).advise([(first, [0])])["status"] == "unsupported"
    # No memory access after truncation; validate fstat before mincore.
    os.truncate(other.mm.filename, 1)
    assert advisor(backend).advise([(other, [0])])["status"] == "unsupported"
    assert not backend.events


def test_empty_selection(readers):
    backend = Backend()
    assert advisor(backend).advise([(readers(), [])])["status"] == "empty"
    assert not backend.events


def test_file_validation_shared_per_call_not_retained(readers, monkeypatch):
    reader = readers()
    original = os.fstat
    calls = []
    def counted(fd):
        calls.append(fd)
        return original(fd)
    monkeypatch.setattr(advice.os, "fstat", counted)
    helper = advisor()
    for _ in range(2):
        assert helper.advise([(reader, [0]), (reader, [1])])["status"] == "advised"
    assert len(calls) == 2


def test_factory_unsupported_platform_does_not_construct_backend(monkeypatch):
    monkeypatch.setattr(advice.sys, "platform", "linux")
    def forbidden():
        pytest.fail("unsupported platform loaded Darwin interfaces")
    monkeypatch.setattr(advice, "_DarwinBackend", forbidden)
    assert advice.SelectedPageReadAdvisor.create() is None


def test_factory_syscall_unavailable(monkeypatch):
    monkeypatch.setattr(advice.sys, "platform", "darwin")
    def unavailable():
        raise AttributeError("mincore")
    monkeypatch.setattr(advice, "_DarwinBackend", unavailable)
    assert advice.SelectedPageReadAdvisor.create() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin mincore/F_RDADVISE ABI")
def test_real_small_readonly_memmap_no_forced_cold_pages(readers):
    reader = readers()
    helper = advice.SelectedPageReadAdvisor.create()
    assert helper is not None
    result = helper.advise([(reader, [0, 7])])
    assert result["status"] in {"resident", "advised"}
    assert result["pages"] == result["resident"] + result["hinted"]
    assert result["pages"] == 1
    expected = np.array([[(19 + row * 32 + col) % 251 for col in range(32)] for row in [0, 7]], dtype=np.uint8)
    np.testing.assert_array_equal(reader.mm[[0, 7]], expected)
