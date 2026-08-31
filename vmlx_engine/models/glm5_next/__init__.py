# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned GLM-5.3-Flash (glm5_next) runtime package."""

from vmlx_engine.models.glm5_next.register import (
    glm5_next_runtime_available,
    glm5_next_vlm_runtime_available,
    register_glm5_next_runtime,
    register_glm5_next_vlm_runtime,
)

__all__ = [
    "glm5_next_runtime_available",
    "glm5_next_vlm_runtime_available",
    "register_glm5_next_runtime",
    "register_glm5_next_vlm_runtime",
]
