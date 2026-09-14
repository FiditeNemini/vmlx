"""Native Qwen4 QSA sparse-GQA Metal kernel for MTPLX.

Optional: MTPLX imports this package defensively. When the extension was
never built the import fails and ``mtplx/kernels/qsa_prefill_direct.py``
reports the lane unsupported without raising.

See NOTICE for provenance (oMLX PR #3244) and the license set.
"""

from ._ext import (  # noqa: F401
    BUILT_AGAINST_MLX,
    BUILT_AGAINST_NANOBIND,
    METAL_LIBRARY,
    abi_probe,
    qwen4_qsa_sparse_gqa_attention,
    qwen4_qsa_sparse_gqa_dense_scores,
)

__all__ = [
    "BUILT_AGAINST_MLX",
    "BUILT_AGAINST_NANOBIND",
    "METAL_LIBRARY",
    "abi_probe",
    "qwen4_qsa_sparse_gqa_attention",
    "qwen4_qsa_sparse_gqa_dense_scores",
]

# Preserve legacy loading against an older local binary. Only the experimental
# consumer requires the new symbol and disables itself when it is absent.
from . import _ext
if hasattr(_ext, "qwen4_qsa_sparse_gqa_attention_nax"):
    qwen4_qsa_sparse_gqa_attention_nax = _ext.qwen4_qsa_sparse_gqa_attention_nax
    __all__.append("qwen4_qsa_sparse_gqa_attention_nax")
