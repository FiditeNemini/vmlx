"""Shape-gated affine MoE gate/up fusion for single-token decode.

The production Qwen4-Exp and GLM5-Next affine bundles issue two selected-
expert ``gather_qmm`` calls before SwiGLU on every routed-MoE layer.  This
module reads the existing gate/up tensors in place and produces the activated
rows with one Metal dispatch. Compatible GLM q2/g128 down projections also
reduce all selected experts directly into the routed output; mixed q3 layers
retain MLX down/reduction. No path duplicates the expert payload.

Only source-audited production geometries are accepted.  Registration is
atomic per family.  Qwen4-Exp is enabled by default after exact-bundle AR/MTP,
Electron, Chat Completions, Responses, image, video, and tool-continuation
proof; its environment flag remains an explicit opt-out.  GLM5-Next remains
opt-in until its own equivalent gate is complete.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_CONFIG_ATTR = "_vmlx_affine_moe_pair_config"
_STATUS: dict[str, dict[str, Any]] = {}
_FIRST_FAST_CALL: set[str] = set()
_FIRST_FALLBACK: set[str] = set()


@dataclass(frozen=True)
class _PairConfig:
    family: str
    hidden: int
    intermediate: int
    top_k: int
    bits: int
    group_size: int
    clamp_limit: float | None
    fuse_down: bool = False

    @property
    def packed_per_row(self) -> int:
        return self.hidden * self.bits // 32

    @property
    def groups_per_row(self) -> int:
        return self.hidden // self.group_size

    @property
    def values_per_word(self) -> int:
        return 32 // self.bits

    @property
    def words_per_group(self) -> int:
        return self.group_size // self.values_per_word

    @property
    def down_packed_per_row(self) -> int:
        return self.intermediate * self.bits // 32

    @property
    def down_groups_per_row(self) -> int:
        return self.intermediate // self.group_size


_FAMILY_CONTRACTS = {
    "qwen4_exp": {
        "hidden": 2560,
        "intermediate": 640,
        "top_k": 10,
        "layouts": {(2, 64), (4, 64)},
        "clamp_limit": None,
    },
    "glm5_next": {
        "hidden": 4096,
        "intermediate": 2048,
        "top_k": 8,
        "layouts": {(2, 128)},
        "clamp_limit": 10.0,
    },
}

_DEFAULT_ENABLED = {
    "qwen4_exp": True,
    "glm5_next": False,
}


def _requested(family: str) -> bool:
    env = {
        "qwen4_exp": "VMLX_QWEN4_FUSED_MOE_PAIR",
        "glm5_next": "VMLX_GLM5_FUSED_MOE_PAIR",
    }[family]
    value = os.environ.get(env)
    if value is None:
        return _DEFAULT_ENABLED[family]
    value = value.strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def _projection_reason(projection: Any, *, hidden: int, intermediate: int) -> str | None:
    if projection.__class__.__name__ != "QuantizedSwitchLinear":
        return f"class={projection.__class__.__name__}"
    bits = int(getattr(projection, "bits", -1))
    group_size = int(getattr(projection, "group_size", -1))
    if bits <= 0 or 32 % bits:
        return f"bits={bits} does not pack evenly into uint32"
    if group_size <= 0 or hidden % group_size:
        return f"group_size={group_size} is incompatible with hidden={hidden}"
    if group_size % (32 // bits):
        return f"group_size={group_size} does not contain whole packed words"
    if str(getattr(projection, "mode", "affine")) != "affine":
        return f"mode={getattr(projection, 'mode', None)}"
    if int(projection.input_dims) != hidden:
        return f"input_dims={projection.input_dims}"
    if int(projection.output_dims) != intermediate:
        return f"output_dims={projection.output_dims}"
    if projection.weight.dtype != mx.uint32:
        return f"weight_dtype={projection.weight.dtype}"
    metadata_dtypes = (mx.float16, mx.bfloat16)
    if projection.scales.dtype not in metadata_dtypes:
        return f"scale_dtype={projection.scales.dtype}"
    if (
        projection.biases is None
        or projection.biases.dtype not in metadata_dtypes
    ):
        return f"bias_dtype={getattr(projection.biases, 'dtype', None)}"
    if projection.biases.dtype != projection.scales.dtype:
        return (
            "affine metadata dtypes differ: "
            f"scales={projection.scales.dtype} biases={projection.biases.dtype}"
        )
    if "bias" in projection:
        return "post-matmul bias is unsupported"
    if projection.weight.ndim != 3 or projection.scales.ndim != 3:
        return "affine expert tensors must be rank three"
    if projection.biases.shape != projection.scales.shape:
        return "affine scale/bias shapes differ"
    if tuple(projection.weight.shape[:2]) != tuple(projection.scales.shape[:2]):
        return "packed weight and affine metadata rows differ"
    if int(projection.weight.shape[2]) != hidden * bits // 32:
        return "packed weight geometry differs from hidden/bits"
    if int(projection.scales.shape[2]) != hidden // group_size:
        return "affine metadata geometry differs from hidden/group_size"
    return None


def _switch_config(switch: Any, family: str) -> _PairConfig:
    contract = _FAMILY_CONTRACTS[family]
    hidden = int(contract["hidden"])
    intermediate = int(contract["intermediate"])
    gate_reason = _projection_reason(
        switch.gate_proj,
        hidden=hidden,
        intermediate=intermediate,
    )
    up_reason = _projection_reason(
        switch.up_proj,
        hidden=hidden,
        intermediate=intermediate,
    )
    if gate_reason is not None or up_reason is not None:
        raise ValueError(f"gate={gate_reason}; up={up_reason}")
    gate_layout = (int(switch.gate_proj.bits), int(switch.gate_proj.group_size))
    up_layout = (int(switch.up_proj.bits), int(switch.up_proj.group_size))
    if gate_layout != up_layout:
        raise ValueError(f"gate/up layouts differ: {gate_layout} != {up_layout}")
    if gate_layout not in contract["layouts"]:
        raise ValueError(f"unsupported {family} gate/up layout {gate_layout}")
    gate_experts = int(switch.gate_proj.weight.shape[0])
    up_experts = int(switch.up_proj.weight.shape[0])
    if gate_experts != up_experts:
        raise ValueError(f"gate/up expert counts differ: {gate_experts} != {up_experts}")
    if gate_experts < int(contract["top_k"]):
        raise ValueError(
            f"expert count {gate_experts} is smaller than top_k={contract['top_k']}"
        )
    clamp_limit = contract["clamp_limit"]
    if clamp_limit is None:
        if switch.activation.__class__.__name__ != "SwiGLU":
            raise ValueError(
                f"unexpected qwen activation {switch.activation.__class__.__name__}"
            )
    elif float(getattr(switch.activation, "_limit", -1.0)) != float(clamp_limit):
        raise ValueError("GLM clamped SwiGLU limit differs from 10")
    down_reason = _projection_reason(
        switch.down_proj,
        hidden=intermediate,
        intermediate=hidden,
    )
    down_layout = (
        int(getattr(switch.down_proj, "bits", -1)),
        int(getattr(switch.down_proj, "group_size", -1)),
    )
    return _PairConfig(
        family=family,
        hidden=hidden,
        intermediate=intermediate,
        top_k=int(contract["top_k"]),
        bits=gate_layout[0],
        group_size=gate_layout[1],
        clamp_limit=clamp_limit,
        fuse_down=(
            family == "glm5_next"
            and down_reason is None
            and down_layout == gate_layout == (2, 128)
        ),
    )


def _pair_source(config: _PairConfig) -> str:
    clamp = ""
    if config.clamp_limit is not None:
        limit = float(config.clamp_limit)
        clamp = f"""
        g = metal::min(g, {limit:.1f}f);
        u = metal::clamp(u, -{limit:.1f}f, {limit:.1f}f);
"""
    return f"""
        uint tid = thread_position_in_grid.x;
        uint sgid = tid / 32u;
        uint lane = thread_index_in_simdgroup;
        uint route = sgid / {config.intermediate}u;
        uint out_d = sgid % {config.intermediate}u;
        if (route >= {config.top_k}u) return;

        uint expert = (uint)expert_ids[route];
        size_t row = (size_t)expert * {config.intermediate}u + out_d;
        const device uint32_t* gate_row = gate_weight + row * {config.packed_per_row}u;
        const device uint32_t* up_row = up_weight + row * {config.packed_per_row}u;
        size_t meta_row = row * {config.groups_per_row}u;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        for (uint word_idx = lane; word_idx < {config.packed_per_row}u; word_idx += 32u) {{
            uint group = word_idx / {config.words_per_group}u;
            float gate_scale = (float)gate_scales[meta_row + group];
            float gate_bias = (float)gate_biases[meta_row + group];
            float up_scale = (float)up_scales[meta_row + group];
            float up_bias = (float)up_biases[meta_row + group];
            uint32_t gate_packed = gate_row[word_idx];
            uint32_t up_packed = up_row[word_idx];
            uint input_base = word_idx * {config.values_per_word}u;
            float gate_quant_dot = 0.0f;
            float up_quant_dot = 0.0f;
            float input_sum = 0.0f;
            for (uint item = 0; item < {config.values_per_word}u; ++item) {{
                float value = (float)x[input_base + item];
                input_sum += value;
                uint shift = {config.bits}u * item;
                gate_quant_dot += value * (float)((gate_packed >> shift) & {(1 << config.bits) - 1}u);
                up_quant_dot += value * (float)((up_packed >> shift) & {(1 << config.bits) - 1}u);
            }}
            gate_acc += gate_scale * gate_quant_dot + gate_bias * input_sum;
            up_acc += up_scale * up_quant_dot + up_bias * input_sum;
        }}
        gate_acc = simd_sum(gate_acc);
        up_acc = simd_sum(up_acc);
        if (lane == 0u) {{
            float g = (float)((T)gate_acc);
            float u = (float)((T)up_acc);
{clamp}
            float activated = g / (1.0f + metal::exp(-g)) * u;
            output[(size_t)route * {config.intermediate}u + out_d] = (T)activated;
        }}
"""


@lru_cache(maxsize=8)
def _pair_kernel(config: _PairConfig):
    return mx.fast.metal_kernel(
        name=(
            f"vmlx_{config.family}_q{config.bits}g{config.group_size}_"
            "selected_pair_swiglu"
        ),
        input_names=[
            "x",
            "gate_weight",
            "gate_scales",
            "gate_biases",
            "up_weight",
            "up_scales",
            "up_biases",
            "expert_ids",
        ],
        output_names=["output"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_pair_source(config),
    )


def _run_pair(
    switch: Any,
    config: _PairConfig,
    x: mx.array,
    indices: mx.array,
) -> mx.array:
    kernel = _pair_kernel(config)
    output = kernel(
        inputs=[
            x.reshape(-1),
            switch.gate_proj.weight,
            switch.gate_proj.scales,
            switch.gate_proj.biases,
            switch.up_proj.weight,
            switch.up_proj.scales,
            switch.up_proj.biases,
            indices.reshape(-1).astype(mx.uint32),
        ],
        template=[("T", x.dtype)],
        grid=(32 * config.intermediate * config.top_k, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(config.top_k, config.intermediate)],
        output_dtypes=[x.dtype],
    )[0]
    return output.reshape(*x.shape[:-1], config.top_k, 1, config.intermediate)


def _down_source(config: _PairConfig) -> str:
    return f"""
        uint tid = thread_position_in_grid.x;
        uint out_d = tid / 32u;
        uint lane = thread_index_in_simdgroup;
        if (out_d >= {config.hidden}u) return;

        float weighted = 0.0f;
        for (uint route = 0u; route < {config.top_k}u; ++route) {{
            uint expert = (uint)expert_ids[route];
            size_t row = (size_t)expert * {config.hidden}u + out_d;
            const device uint32_t* packed_row =
                down_weight + row * {config.down_packed_per_row}u;
            size_t meta_row = row * {config.down_groups_per_row}u;
            float acc = 0.0f;
            for (uint word_idx = lane;
                 word_idx < {config.down_packed_per_row}u;
                 word_idx += 32u) {{
                uint group = word_idx / {config.words_per_group}u;
                float scale = (float)down_scales[meta_row + group];
                float bias = (float)down_biases[meta_row + group];
                uint32_t packed = packed_row[word_idx];
                uint input_base = route * {config.intermediate}u +
                    word_idx * {config.values_per_word}u;
                float quant_dot = 0.0f;
                float input_sum = 0.0f;
                for (uint item = 0u; item < {config.values_per_word}u; ++item) {{
                    float value = (float)activated[input_base + item];
                    input_sum += value;
                    uint shift = {config.bits}u * item;
                    quant_dot += value *
                        (float)((packed >> shift) & {(1 << config.bits) - 1}u);
                }}
                acc += scale * quant_dot + bias * input_sum;
            }}
            acc = simd_sum(acc);
            weighted += (float)route_scores[route] * acc;
        }}
        if (lane == 0u) output[out_d] = (T)weighted;
"""


@lru_cache(maxsize=4)
def _down_kernel(config: _PairConfig):
    return mx.fast.metal_kernel(
        name=(
            f"vmlx_{config.family}_q{config.bits}g{config.group_size}_"
            "weighted_down"
        ),
        input_names=[
            "activated",
            "down_weight",
            "down_scales",
            "down_biases",
            "expert_ids",
            "route_scores",
        ],
        output_names=["output"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=_down_source(config),
    )


def _run_weighted_down(
    switch: Any,
    config: _PairConfig,
    activated: mx.array,
    indices: mx.array,
    scores: mx.array,
) -> mx.array:
    projection = switch.down_proj
    output = _down_kernel(config)(
        inputs=[
            activated.reshape(config.top_k, config.intermediate),
            projection.weight,
            projection.scales,
            projection.biases,
            indices.reshape(-1).astype(mx.uint32),
            scores.reshape(-1).astype(activated.dtype),
        ],
        template=[("T", activated.dtype)],
        grid=(32 * config.hidden, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(config.hidden,)],
        output_dtypes=[activated.dtype],
    )[0]
    return output.reshape(*scores.shape[:-1], config.hidden)


def affine_moe_pair_activation(
    switch: Any,
    x: mx.array,
    indices: mx.array,
) -> tuple[mx.array | None, bool]:
    """Return fused activated expert rows when the registered shape owns it."""

    config = getattr(switch, _CONFIG_ATTR, None)
    if config is None or bool(getattr(switch, "training", False)):
        return None, False
    reason = None
    if x.dtype != mx.float16:
        reason = f"activation_dtype={x.dtype}"
    elif x.ndim not in (2, 3) or int(x.size) != int(x.shape[-1]):
        reason = f"non_decode_shape={tuple(x.shape)}"
    elif int(x.shape[-1]) != config.hidden:
        reason = f"hidden={int(x.shape[-1])}"
    elif indices.ndim != x.ndim or tuple(indices.shape[:-1]) != tuple(x.shape[:-1]):
        reason = f"route_shape={tuple(indices.shape)}"
    elif int(indices.shape[-1]) != config.top_k:
        reason = f"top_k={int(indices.shape[-1])}"
    if reason is not None:
        if config.family not in _FIRST_FALLBACK:
            logger.info("%s affine MoE pair fusion used stock path: %s", config.family, reason)
            _FIRST_FALLBACK.add(config.family)
        return None, False
    try:
        output = _run_pair(switch, config, x, indices)
        if config.family not in _FIRST_FAST_CALL:
            mx.eval(output)
            logger.info(
                "%s affine MoE pair fusion active: q%d/g%d hidden=%d intermediate=%d top_k=%d",
                config.family,
                config.bits,
                config.group_size,
                config.hidden,
                config.intermediate,
                config.top_k,
            )
            _FIRST_FAST_CALL.add(config.family)
            _STATUS.setdefault(config.family, {})["observed_calls"] = 1
        return output, True
    except Exception as exc:
        if hasattr(switch, _CONFIG_ATTR):
            delattr(switch, _CONFIG_ATTR)
        logger.exception(
            "%s affine MoE pair fusion failed; disabling this module", config.family
        )
        _STATUS.setdefault(config.family, {})["last_runtime_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return None, False


def affine_moe_routed_output(
    switch: Any,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
) -> tuple[mx.array | None, bool]:
    """Return the weighted routed output when a registered decode path owns it."""

    activated, pair_fused = affine_moe_pair_activation(switch, x, indices)
    if not pair_fused or activated is None:
        return None, False
    config = getattr(switch, _CONFIG_ATTR)
    if config.fuse_down:
        try:
            return _run_weighted_down(
                switch, config, activated, indices, scores
            ), True
        except Exception as exc:
            if hasattr(switch, _CONFIG_ATTR):
                delattr(switch, _CONFIG_ATTR)
            logger.exception(
                "%s affine weighted-down fusion failed; disabling this module",
                config.family,
            )
            _STATUS.setdefault(config.family, {})["last_down_runtime_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return None, False
    selected = switch.down_proj(activated, indices).squeeze(-2)
    return (selected * scores[..., None].astype(selected.dtype)).sum(axis=-2), True


def install_affine_moe_pair_decode(model: Any, *, family: str) -> int:
    """Register every compatible SwitchGLU atomically for one model family."""

    if family not in _FAMILY_CONTRACTS:
        raise ValueError(f"unsupported affine MoE pair family {family}")
    if not _requested(family):
        _STATUS[family] = {"installed": 0, "reason": "disabled via env"}
        return 0
    from mlx_lm.models.switch_layers import SwitchGLU

    modules = [
        module
        for _name, module in model.named_modules()
        if isinstance(module, SwitchGLU)
    ]
    if not modules:
        _STATUS[family] = {"installed": 0, "reason": "no SwitchGLU modules"}
        return 0
    accepted: list[tuple[Any, _PairConfig]] = []
    rejected: list[str] = []
    for module in modules:
        try:
            accepted.append((module, _switch_config(module, family)))
        except (AttributeError, TypeError, ValueError) as exc:
            rejected.append(str(exc))
    if rejected:
        for module in modules:
            if hasattr(module, _CONFIG_ATTR):
                delattr(module, _CONFIG_ATTR)
        _STATUS[family] = {
            "installed": 0,
            "reason": f"incompatible modules: {rejected[:3]}",
        }
        return 0
    for module, config in accepted:
        setattr(module, _CONFIG_ATTR, config)
    layouts = sorted({(config.bits, config.group_size) for _, config in accepted})
    full_down_modules = sum(config.fuse_down for _, config in accepted)
    _STATUS[family] = {
        "installed": len(accepted),
        "reason": None,
        "layouts": layouts,
        "full_down_modules": full_down_modules,
    }
    logger.info(
        "%s affine MoE pair fusion registered for %d modules; layouts=%s; "
        "full_down=%d",
        family,
        len(accepted),
        layouts,
        full_down_modules,
    )
    return len(accepted)


def affine_moe_pair_status(family: str | None = None) -> dict[str, Any]:
    if family is not None:
        return dict(_STATUS.get(family, {}))
    return {key: dict(value) for key, value in _STATUS.items()}


__all__ = [
    "affine_moe_pair_activation",
    "affine_moe_routed_output",
    "affine_moe_pair_status",
    "install_affine_moe_pair_decode",
]
