# SPDX-License-Identifier: Apache-2.0
"""A transient read failure must not delete the L2 chain; corruption still must.

`_queue_index_cleanup` does not remove one entry — it walks every descendant
with `ancestry_known=1` and unlinks their payloads too. So misclassifying an
fd-exhaustion window as corruption destroys a healthy chain and turns a warm
long-context read back into a cold prefill.

Both directions are asserted here, because the fix is only correct if a genuinely
corrupt block is still evicted.
"""

import mlx.core as mx
import pytest

from vmlx_engine.block_disk_store import BlockDiskStore


@pytest.fixture
def store(tmp_path):
    return BlockDiskStore(cache_dir=str(tmp_path / "l2"))


TRANSIENT = [
    ("mlx_open_failure", RuntimeError(
        "[load_safetensors] Failed to open file /x/y.safetensors")),
    ("emfile", OSError(24, "Too many open files")),
    ("eio", OSError(5, "Input/output error")),
    ("out_of_memory", MemoryError()),
]

CORRUPT = [
    ("invalid_json_header", RuntimeError(
        "[load_safetensors] Invalid json header length file /x/y.safetensors")),
    ("short_header", RuntimeError(
        "[load_safetensors] The JSON header is 77 bytes long but the file is "
        "only 74 bytes. Perhaps an incomplete download?")),
    ("bad_tensor_map", KeyError("keys")),
    ("bad_shape", ValueError("shape mismatch")),
]


@pytest.mark.parametrize("name,exc", TRANSIENT, ids=[n for n, _ in TRANSIENT])
def test_transient_failure_is_not_treated_as_corruption(store, tmp_path, name, exc):
    readable = tmp_path / "readable.safetensors"
    mx.save_safetensors(str(readable), {"x": mx.zeros((2, 2))})
    assert store._read_failure_is_transient(readable, exc) is True


@pytest.mark.parametrize("name,exc", CORRUPT, ids=[n for n, _ in CORRUPT])
def test_content_failure_on_a_readable_file_is_corruption(store, tmp_path, name, exc):
    readable = tmp_path / "readable.safetensors"
    mx.save_safetensors(str(readable), {"x": mx.zeros((2, 2))})
    assert store._read_failure_is_transient(readable, exc) is False


def test_unopenable_file_is_transient_whatever_the_exception(store, tmp_path):
    """Under fd pressure the probe fails too — same verdict, which is the point."""
    missing = tmp_path / "gone.safetensors"
    assert store._read_failure_is_transient(missing, ValueError("shape mismatch")) is True


def test_a_real_corrupt_file_still_reports_corruption(store, tmp_path):
    """End-to-end against MLX itself, not a hand-written exception."""
    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"not a safetensors file at all")
    try:
        mx.load(str(corrupt))
    except Exception as exc:  # noqa: BLE001 - the point is whatever MLX raises
        assert store._read_failure_is_transient(corrupt, exc) is False
    else:
        pytest.fail("expected mx.load to reject a corrupt file")


def test_a_real_missing_file_reports_transient(store, tmp_path):
    absent = tmp_path / "absent.safetensors"
    try:
        mx.load(str(absent))
    except Exception as exc:  # noqa: BLE001
        assert store._read_failure_is_transient(absent, exc) is True
    else:
        pytest.fail("expected mx.load to fail on a missing file")
