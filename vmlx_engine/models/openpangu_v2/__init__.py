# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned openPangu-2.0-Flash (openpangu_v2) runtime package."""

from vmlx_engine.models.openpangu_v2.register import (
    openpangu_v2_runtime_available,
    register_openpangu_v2_runtime,
)

__all__ = [
    "openpangu_v2_runtime_available",
    "register_openpangu_v2_runtime",
]
