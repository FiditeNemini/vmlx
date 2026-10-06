"""First-use codebook headers retain exact bytes without fitting unused widths."""
import subprocess
import sys

import pytest


def test_cold_header_avoids_fit_and_preserves_generated_bytes():
    pytest.importorskip("mlx.core")
    # A fresh process avoids hiding a first-use fit behind another test's cache.
    subprocess.run([sys.executable, "-c", """
import hashlib
from vmlx_engine.jangh import contract, format, kernels

def unexpected_fit(*args, **kwargs):
    raise AssertionError("Cold header must not fit a codebook")

format._mse_levels = unexpected_fit
assert set(contract.CUBIC_PARAMS) == {2, 3, 4, 6, 8}  # 6/8-bit added by e84dc16f; header bytes unchanged
header = kernels._cb_header().encode()
assert hashlib.sha256(header).hexdigest() == (
    "1bec7d31ea9f2384fd4f1231adf1d66a4220d021987de267301dd55321ad05be"
)
"""], check=True)
