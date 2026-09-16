# SPDX-License-Identifier: Apache-2.0
"""Pure GLM native-cache policy shared by runtime storage and admission."""
import os

GLM5_MLA_CAPACITY_TOKENS = 2048


def glm5_native_ssd_requested(model) -> bool:
    """Select native MLLM storage without probing unrelated model families.

    Configuration is only the cheap candidate filter. The scheduler must still
    inspect every actual cache object and enforce its single-request contract.
    Explicit values retain the original experimental switch semantics.
    """
    value = os.environ.get("VMLX_GLM5_NATIVE_SSD")
    if value is not None:
        return value == "1"
    config = getattr(model, "config", None)
    model_type = (config.get("model_type") if isinstance(config, dict)
                  else getattr(config, "model_type", None))
    return isinstance(model_type, str) and model_type in {"glm5_next", "glm5_next_text"}


def glm5_native_media_ssd_enabled() -> bool:
    """Complete image/video identity is required; retain an explicit opt-out."""
    return os.environ.get("VMLX_GLM5_NATIVE_MEDIA_SSD", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }


def glm5_mla_absorb_enabled() -> bool:
    return os.environ.get(
        "VMLINUX_GLM5_MLA_ABSORB",
        os.environ.get("VMLX_GLM5_MLA_ABSORB", "1"),
    ).strip().lower() in {"1", "true", "yes", "on"}


def glm5_dsa_bf16_state_enabled() -> bool:
    return os.environ.get("VMLX_GLM5_DSA_BF16", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
