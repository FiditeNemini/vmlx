# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned Qwen 3.5 / Qwen 3.5 MoE VLM runtime.

Vendored to fix an mlx-vlm bug that forces routing gates (``mlp.gate`` and
``shared_expert_gate``) to quantize at bits=8, group_size=64 via
``LanguageModel.quant_predicate``. JANG affine bundles (Ornith 9B/35B/397B,
JANG_4M/6M/1L) ship those gates as **float16 without ``.scales``**. When the
upstream predicate wraps them in ``nn.QuantizedLinear``, the float16 weight
survives inside a Quantized shell — and every decode step crashes with:

    [quantized_matmul] The weight matrix should be uint32 but received float16

The vMLX vendor keeps the gates as plain ``nn.Linear`` (matching mlx-lm's
``Qwen3NextSparseMoeBlock``), so both the standard-load path (text-only via
mlx-lm) and the JANG-VL-bundle / Smelt path (via mlx-vlm) agree on the router
representation.

Registration is idempotent and installs the modules into
``mlx_vlm.models.qwen3_5`` and ``mlx_vlm.models.qwen3_5_moe`` — the vendored
files use relative ``from ..base import`` / ``from ..cache import`` /
``from ..qwen3_vl import`` imports, which continue to resolve to upstream
mlx-vlm because we live INSIDE that namespace at runtime.
"""

from .register import register_qwen3_5_family_runtime

__all__ = ["register_qwen3_5_family_runtime"]
