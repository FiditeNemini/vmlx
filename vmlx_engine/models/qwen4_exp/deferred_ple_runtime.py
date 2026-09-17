"""Opt-in productive-AR packed PLE scheduling with native cache isolation."""
from __future__ import annotations

import copy
import logging
import os
import sys

import mlx.core as mx

from .deferred_ple import Declined, UnsafeDeferredEvaluation
from .packed_ple import PackedDeferredPLE

logger = logging.getLogger(__name__)
_LOGGED = False


def _clone_value(value, frontier_type):
    # __copy__ returns a distinct C++ array handle sharing immutable Data.
    # MLX Python __setitem__ overwrites THAT handle's descriptor, never the
    # original handle. Strong references also keep source buffers non-donatable.
    if isinstance(value, mx.array):
        return copy.copy(value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is list:
        return [_clone_value(x, frontier_type) for x in value]
    if type(value) is tuple:
        return tuple(_clone_value(x, frontier_type) for x in value)
    if type(value) is dict:
        if any(type(k) not in (str, int) for k in value):
            raise Declined("Deferred PLE unknown native cache dictionary key")
        return {k: _clone_value(v, frontier_type) for k, v in value.items()}
    if type(value) is frontier_type:
        clone = copy.copy(value)
        for key in frontier_type.__slots__:
            setattr(clone, key, _clone_value(getattr(value, key), frontier_type))
        return clone
    raise Declined(f"Deferred PLE unknown mutable cache field: {type(value).__name__}")


def _helpers_ready(layers, cache=None, activation_dtype=None):
    # Read real helper engagement receipts; never manufacture readiness and
    # never delete first-use selfchecks. If a context transition enables a
    # previously unvisited helper, this forward remains stock until ready.
    from vmlx_engine.metal import affine_moe_decode as scalar
    from vmlx_engine.metal import affine_moe_pair_decode as pair
    from vmlx_engine.metal import qwen4_exact_down as down
    from vmlx_engine.metal import qwen4_qsa_mask as mask
    from vmlx_engine.metal import qwen4_qsa_score_reduce as score
    from vmlx_engine.metal import sparse_merge_topk as merge
    from vmlx_engine.metal import qwen4_sparse_decode as sparse
    for layer_index, layer in enumerate(layers):
        switch = layer.mlp.switch_mlp
        pc = getattr(switch, pair._CONFIG_ATTR, None)
        full_fused = (getattr(switch, "_vmlx_qwen4_q4g64_fused_ok", False)
                      and layer.mlp.top_k == 10
                      and switch.gate_proj.input_dims == 2560
                      and (activation_dtype is None or activation_dtype in (mx.float16, mx.bfloat16)))
        pair_eligible = (pc is not None and not full_fused
                         and not getattr(switch, "training", False)
                         and pc.hidden == switch.gate_proj.input_dims
                         and pc.top_k == layer.mlp.top_k
                         and (activation_dtype is None or activation_dtype in (mx.float16, mx.bfloat16)))
        if pair_eligible:
            if pc.family not in pair._FIRST_FAST_CALL:
                return False
            projection = switch.down_proj
            q30_eligible = (pc.intermediate == 640 and pc.hidden == 2560 and pc.top_k == 10
                            and (activation_dtype is None or activation_dtype == mx.float16)
                            and not getattr(projection, "training", False)
                            and down._projection_reason(projection, hidden=640, intermediate=2560) is None
                            and projection.group_size in (32, 64)
                            and projection.scales.dtype == mx.float16)
            if q30_eligible and down._ENABLED and not (down._OBSERVED or down._FAILED):
                return False
        elif not full_fused and not getattr(switch, "_vmlx_qwen4_exact_gate_up_ok", False):
            sc = scalar._CONFIGS.get(switch)
            if sc is not None and not sc.first_call_evaluated:
                return False
        if layer.is_linear:
            continue
        attn, index = layer.self_attn, layer.self_attn.indexer
        # Do not wait forever for a helper that cannot run at this context.
        # Check the incoming token's POST-append extent, exactly as attention.
        tokens = None if cache is None else int(cache[layer_index].offset) + 1
        pools = None if tokens is None else tokens // index.compress_ratio
        # _mask_from_payload returns before ALL score/selection/mask helpers
        # when every completed block is visible. Match the actual metadata,
        # not just the current 4S token thresholds. These helpers expose no
        # side-effect-free tensor admission API; mirror their scalar guards.
        selection_needed = pools is None or pools > index.block_topk
        sparse_eligible = (
            selection_needed and (tokens is None or 65537 <= tokens <= 131072)
            and not getattr(attn, "training", False)
            and (attn.num_heads, attn.num_kv_heads, attn.head_dim) == (24, 2, 256)
            and attn.scale == 0.0625 and not os.environ.get("MLX_SDPA_BLOCKS")
            and sparse._hardware_allowed())
        mask_eligible = (selection_needed and index.compress_ratio == 4
                         and (tokens is None or 2048 < tokens <= 131072))
        score_eligible = selection_needed and index.n_heads == 4 and index.head_dim > 0
        merge_eligible = (selection_needed and index.block_topk == merge._K
                          and (pools is None or 2048 < pools < 2**32)
                          and merge._compatible_sort_version())
        if (attn._sparse_ar_decode and sparse_eligible
                and (activation_dtype is None or activation_dtype == mx.float16)
                and not attn._sparse_ar_observed):
            return False
        if (index._fused_block_mask and mask_eligible
                and not (mask._FAILED or (1, 1) in mask._OBSERVED)):
            return False
        if (index._score_reduce and not index._fused_score_decode
                and score_eligible
                and not (score._failed or score._observed)):
            return False
        if index._merge_select and merge_eligible and not (merge._FAILED or
                (merge._OBSERVED and {"tile", "merge"} <= merge._VERIFIED_VARIANTS)):
            return False
    return True


def _make_shadows(layers, cache, arrays_type, sparse_type, frontier_type):
    shadows = []
    for layer, original in zip(layers, cache):
        expected = arrays_type if layer.is_linear else sparse_type
        if type(original) is not expected:
            raise Declined("Deferred PLE refuses batch/proxy/quantized/unknown caches")
        if any(getattr(original, name, None) is not None for name in
               ("rollback_state", "rollback_aux_state", "rollback_aux_to",
                "prefill_checkpoint_aux_states", "prefill_checkpoint_states")):
            raise Declined("Deferred PLE refuses active rollback/checkpoint caches")
        shadow = copy.copy(original)
        shadow.__dict__ = _clone_value(original.__dict__, frontier_type)
        shadows.append(shadow)
    return shadows


def _run_shadow_transaction(forward, originals, shadows, scope):
    """No replay: all errors discard shadows without touching native owners."""
    try:
        with scope:
            output = forward(shadows)
            scope.flush()
    except UnsafeDeferredEvaluation as exc:
        raise RuntimeError("Deferred PLE aborted before an unsafe pending-leaf evaluation") from exc
    for original, shadow in zip(originals, shadows):
        original.__dict__ = shadow.__dict__
    return output


def productive_ar_forward(owner, inputs, lm_kwargs):
    """Called exclusively inside MLLMBatchGenerator._step's affine AR scope."""
    lm = owner.language_model

    def stock():
        return lm(inputs, **lm_kwargs)

    if os.environ.get("VMLX_QWEN4_DEFER_PACKED_PLE", "0") != "1":
        return stock()
    if (mx.__version__ != "0.32.2" or mx.default_device() != mx.gpu
            or tuple(inputs.shape) != (1, 1)
            or owner._model_type not in {"qwen4_exp", "qwen4_exp_text"}
            or getattr(owner, "_decode_trace", False)):
        return stock()
    requests = getattr(getattr(owner, "active_batch", None), "requests", ())
    if (len(requests) != 1
            or getattr(requests[0], "_native_mtp_state", None) is not None
            or getattr(requests[0], "_native_mtp_ar_tier", None) is not None):
        return stock()
    from mlx_lm.models.cache import ArraysCache
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.native_mtp_prompt_priming import capture_requested
    model = getattr(lm, "model", None)
    module_name = type(model).__module__
    if module_name not in ("vmlx_engine.models.qwen4_exp.language",
                           "mlx_vlm.models.qwen4_exp.language"):
        return stock()
    source = sys.modules.get(module_name)
    if source is None or type(model) is not source.Qwen4ExpTextModel:
        return stock()
    frontier_type = source._QSAPooledFrontier
    if (source._layer_profile_enabled(inputs) or capture_requested(lm)
            or os.environ.get("VMLX_DIAG_RESTORE_FINGERPRINT", "0").lower()
               not in ("", "0", "false", "off", "no")
            or getattr(lm, "training", False)):
        return stock()
    cache = lm_kwargs.get("cache")
    if (type(cache) is not list or len(cache) != len(model.layers)
            or set(lm_kwargs) - {"cache", "position_ids"}):
        return stock()
    try:
        shadows = _make_shadows(model.layers, cache, ArraysCache,
                                MiniMaxM3SparseCache, frontier_type)
        embedding = model.embed_tokens
        metadata = getattr(embedding, "scales", None)
        activation_dtype = metadata.dtype if metadata is not None else embedding.weight.dtype
        if not _helpers_ready(model.layers, cache, activation_dtype):
            return stock()
        plans = [(layer.ple, shadow) for layer, shadow in zip(model.layers, shadows)
                 if layer.ple is not None]
        scope = PackedDeferredPLE(plans)  # all layout/adoption checks pre-build
    except Declined:
        return stock()
    kwargs = dict(lm_kwargs, cache=shadows)
    # No try/stock fallback: errors abandon ONLY shadows + pending leaves.
    # The caller's native objects have not been modified and stay reusable.
    output = _run_shadow_transaction(lambda _shadows: lm(inputs, **kwargs),
                                     cache, shadows, scope)
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info("Deferred PLE packed PLE active: AR B1S1 layers=%d exact_layout "
                    "scope=full_graph host_fill=before_submit shadow_cache=true",
                    len(plans))
    return output
