# SPDX-License-Identifier: Apache-2.0
"""Pure GLM native-cache policy shared by runtime storage and admission."""
import os

GLM5_MLA_CAPACITY_TOKENS = 2048


def glm5_mla_absorb_enabled() -> bool:
    return os.environ.get(
        "VMLINUX_GLM5_MLA_ABSORB",
        os.environ.get("VMLX_GLM5_MLA_ABSORB", "1"),
    ).strip().lower() in {"1", "true", "yes", "on"}


def glm5_dsa_bf16_state_enabled() -> bool:
    return os.environ.get("VMLX_GLM5_DSA_BF16", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
