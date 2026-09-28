# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned Naive-N0.5-Flash (naive_n05_flash) runtime package."""
from vmlx_engine.models.naive_n05_flash.register import (
    naive_n05_flash_runtime_available,
    register_naive_n05_flash_runtime,
)

__all__ = ["naive_n05_flash_runtime_available", "register_naive_n05_flash_runtime"]
