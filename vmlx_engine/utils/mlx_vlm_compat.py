"""Runtime compatibility patches for upstream mlx_vlm bugs.

These are monkey-patches applied once at import time. They wrap known-broken
methods so callers don't need to fork vendored files.

Patches applied
---------------

* Qwen3-VL ``VisionModel.rot_pos_emb`` — upstream types ``grid_thw`` as
  ``mx.array`` but the caller sometimes passes a numpy ``ndarray`` (seen on
  Qwen3.5-35B-A3B bf16, issue #69). ``mx.max(ndarray)`` raises
  ``TypeError: max(): incompatible function arguments``. Coerce on entry.
* Qwen3-VL ``VisionModel.__call__`` — same ``grid_thw`` typing issue when
  ``fast_pos_embed_interpolate`` iterates over a numpy array.
* Qwen3.5/3.6 VL ``Model.sanitize`` — HF-native 3D patch-embed weights can
  arrive as ``(out, channels, temporal, height, width)`` while MLX Conv3D
  expects channels-last ``(out, temporal, height, width, channels)``.
* mlx-vlm ``PromptCacheState`` restore trims cached KV as rank-4 tensors only.
  Qwen3.5/N2 VLM cache tensors can be rank-3 ``(batch, seq, hidden)``; wrap
  them so upstream's rank-4 trim syntax maps back to rank-3 slicing.
* mlx-vlm full-image prefill creates a prompt cache before Qwen/N2 language
  forward, but only primes mRoPE deltas on cached-prefix reuse. Prime the same
  Qwen ``get_rope_index`` state for first full multimodal prefill.
* Qwen3.5/N2 language forward can still reach its nonzero-cache delta branch
  with ``self._rope_deltas`` unset. Use explicit request deltas or recompute
  from Qwen's own ``get_rope_index`` instead of adding a cache offset to None.
  Single-element MLX cache offsets are also normalized to Python ints so the
  precomputed ``_position_ids`` slice path is not skipped by array truthiness.
"""

from __future__ import annotations

import logging
import textwrap

from vmlx_engine.utils.cache_extent import logical_truncate_target

_logger = logging.getLogger(__name__)
_applied = False


def apply() -> None:
    """Apply all mlx_vlm compat patches (idempotent)."""
    global _applied
    _applied = True
    # Register vMLX-owned Qwen 3.5/3.5-MoE VLM runtime FIRST — this replaces
    # mlx_vlm.models.qwen3_5{,_moe} in sys.modules with our vendor whose
    # quant_predicate keeps router gates as float nn.Linear (upstream forces
    # them to bits=8/gs=64, which crashes JANG affine bundles that ship gate
    # weights as float16 without .scales — "weight matrix should be uint32").
    # Must run BEFORE any patches or loader logic that touches those modules.
    try:
        from ..models.qwen3_5_family import register_qwen3_5_family_runtime
        register_qwen3_5_family_runtime()
    except Exception as _exc:
        _logger.warning(
            "mlx_vlm_compat: qwen3_5_family vendor registration failed (%s) — "
            "Ornith / JANG-affine Qwen 3.5 MoE bundles may crash on decode",
            _exc,
        )
    _patch_qwen3_vl_grid_thw()
    _patch_qwen35_patch_embed_layout()
    _patch_lfm2_vl_input_embeddings()
    _patch_qwen3_vl_vision_model_type_allowlist()
    _patch_prompt_cache_rank3_trim()
    _patch_qwen35_mixed_image_video_embeddings()
    # MRoPE none-delta patch (Eric perf-regression suspect 2026-06-27):
    # monkey-patches qwen3_5/qwen3_5_moe LanguageModel.__call__ to add a
    # get_rope_index fallback when rope_deltas is None. If self._rope_deltas
    # isn't primed during prefill, the patched __call__ calls get_rope_index
    # PER DECODE STEP — catastrophic for text-only Ornith 397B
    # (10 tok/s observed vs 20-30 tok/s expected). Gate with
    # VMLX_DISABLE_QWEN35_MROPE_PATCH=1 to A/B test or disable in production.
    import os as _os
    if _os.environ.get("VMLX_DISABLE_QWEN35_MROPE_PATCH", "0") != "1":
        _patch_qwen35_language_mrope_none_delta()
    _patch_gemma4_attention_rope_order()
    _patch_gemma4_proportional_rope_batch_offset()


def _patch_gemma4_proportional_rope_batch_offset() -> None:
    """gemma4 ProportionalRoPE: expand scalar offset to per-row at batch>1.

    mx.fast.rope (mlx 0.31.2) computes rows 1+ INCORRECTLY (garbage or NaN)
    when called with batch>1 and a SCALAR offset at gemma4 full-attention
    shapes (e.g. (B,16,1,128) f16, freqs len 64) — proven by minimal repro
    2026-07-08: identical duplicate rows in, row0 bit-exact vs solo, row1
    NaN/garbage; the SAME call with a per-row offset array [o]*B is bit-exact
    on every row (F16 root cause, task #59). Sliding-layer standard RoPE at
    (B,8,1,256) is unaffected, which is why only full-attention layers (8/48)
    corrupted. Until the Metal kernel is fixed upstream, force the safe
    array-offset code path whenever more than one row is present. Values are
    unchanged — this only dodges the broken scalar kernel path. Idempotent.
    """
    try:
        import mlx.core as _mx
        from mlx_vlm.models.gemma4.rope_utils import ProportionalRoPE as _PRoPE
    except Exception:
        return
    if getattr(_PRoPE.__call__, "_vmlx_batch_offset_patched", False):
        return
    _orig_call = _PRoPE.__call__

    def _batch_safe_call(self, x, offset=0):
        if x.shape[0] > 1 and (
            isinstance(offset, int)
            or (isinstance(offset, _mx.array) and offset.ndim == 0)
        ):
            offset = _mx.full((x.shape[0],), offset, dtype=_mx.int32)
        return _orig_call(self, x, offset=offset)

    _batch_safe_call._vmlx_batch_offset_patched = True
    _PRoPE.__call__ = _batch_safe_call
    _logger.info(
        "mlx_vlm_compat: gemma4 ProportionalRoPE patched — scalar rope offset "
        "expanded to per-row array at batch>1 (mx.fast.rope scalar-offset "
        "batch corruption workaround, task #59/F16)"
    )


def _gemma4_cache_rope_offset(cache):
    """Per-row rope offset for the gemma4 attention patch.

    _step() wraps BatchKVCache in _BatchOffsetSafeCache so Qwen slice ops see
    a scalar int — but consuming that flattened scalar for ROPE gives every
    row the max row's position: joining rows hit the mx.fast.rope scalar-batch
    kernel bug (garbage/NaN, F16) and shorter rows are silently roped at the
    wrong position. Read the TRUE offset from the proxied cache instead —
    Batch* caches carry a per-row mx.array, raw caches a Python int.
    """
    import mlx.core as _mx

    raw = getattr(cache, "_inner", cache).offset
    return raw if isinstance(raw, _mx.array) else _mx.array(raw)


def _patch_gemma4_attention_rope_order() -> None:
    """gemma4: rope queries BEFORE cache.update_and_fetch (batch corruption).

    Upstream ``mlx_vlm.models.gemma4.language.Attention.__call__`` ropes the
    QUERIES only *after* ``cache.update_and_fetch(...)``. Batch caches
    (BatchKVCache.update_and_fetch / BatchRotatingKVCache._update_in_place)
    mutate ``self.offset`` IN PLACE (``self.offset += S`` on an mx.array), so
    the query-rope graph node — built after the mutation — reads offset+1
    while the keys/history were roped at offset: a +1 q/k positional skew on
    EVERY batched decode step. Raw KVCache/RotatingKVCache carry Python-int
    offsets (captured by value) and are immune — which is exactly why serial
    requests were clean and only concurrent joins (which convert the running
    caches to Batch classes, permanently) degenerated into token loops
    ("The capital capital capital…", sweep 2026-07-05, task #76). Sliding
    layers (theta 10k, full-dim rotation) flip the greedy argmax (logits
    maxdiff ~12). Pinning the offset before the update is bit-exact vs the
    raw-cache decode; this reorder was proven token-identical batched vs
    solo. Idempotent; no-ops if upstream is absent.
    """
    try:
        import mlx.core as _mx
        from mlx_vlm.models.gemma4 import language as _g4lang
        from mlx_vlm.models.gemma4.language import (
            scaled_dot_product_attention as _g4_sdpa,
        )
    except Exception:
        return
    _attn_cls = getattr(_g4lang, "Attention", None)
    if _attn_cls is None or getattr(
        _attn_cls.__call__, "_vmlx_rope_order_patched", False
    ):
        return

    # Two upstream shapes exist. Dispatch on the signature so we replicate
    # the installed version faithfully (a mismatch feeds tuples into
    # rms_norm and 500s every request — burned once on 2026-07-05):
    #   new-style: __call__(x, mask, cache, shared_kv=None, offset=None)
    #              -> (out, (keys, values), offset)
    #   old-style: __call__(x, mask, cache) -> out, with an
    #              is_kv_shared_layer state-read branch inline.
    import inspect as _inspect

    _params = _inspect.signature(_attn_cls.__call__).parameters
    _new_style = "shared_kv" in _params

    def _call_new_style(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
            queries = queries.transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(
                    B, L, self.n_kv_heads, self.head_dim
                )
            # Per-row offsets (NOT the proxy's flattened scalar): scalar rope
            # offsets at batch>1 hit the mx.fast.rope row-corruption bug (F16)
            # and mis-position shorter rows. See _gemma4_cache_rope_offset.
            offset = _gemma4_cache_rope_offset(cache) if cache is not None else 0
            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)
            # vMLX reorder: rope queries with the SAME pre-update offset
            # BEFORE the cache mutates it in place.
            queries = queries.transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        output = _g4_sdpa(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values), offset

    def _call_old_style(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        offset = 0
        if self.is_kv_shared_layer and cache is not None:
            state = cache.state
            keys, values = state[0], state[1]
            # Per-row offsets — see _gemma4_cache_rope_offset (F16).
            offset = _gemma4_cache_rope_offset(cache)
            queries = queries.transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)
        else:
            if cache is not None:
                offset = _gemma4_cache_rope_offset(cache)
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(
                    B, L, self.n_kv_heads, self.head_dim
                )
            keys = self.k_norm(keys)
            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)
            # vMLX reorder: rope queries with the SAME pre-update offset
            # BEFORE update_and_fetch mutates cache.offset in place.
            queries = queries.transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, _mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2]:]

        output = _g4_sdpa(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    _patched = _call_new_style if _new_style else _call_old_style
    _patched._vmlx_rope_order_patched = True
    _attn_cls.__call__ = _patched
    _logger.info(
        "mlx_vlm_compat: gemma4 Attention patched (%s-style) — queries roped "
        "before cache.update_and_fetch (batched q/k offset-skew fix, task #76)",
        "new" if _new_style else "old",
    )


class _Rank3KVTrimView:
    """Rank-4 trim facade for rank-3 mlx-vlm KV tensors.

    Upstream mlx-vlm checks ``keys.shape[2]`` and slices
    ``keys[:, :, :prefix_len, :]``. For rank-3 caches the sequence axis is 1,
    so this view exposes a rank-4-looking shape and remaps that slice to
    ``array[:, :prefix_len, :]``. The reported sequence length is one larger
    than the underlying tensor so upstream always assigns the real trimmed
    MLX array back to ``cache.keys``/``cache.values`` before model forward.
    """

    def __init__(self, array):
        self._array = array

    @property
    def shape(self):
        bsz, seq, hidden = self._array.shape
        return (bsz, 1, seq + 1, hidden)

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 4:
            batch_index, _head_index, seq_index, hidden_index = index
            return self._array[batch_index, seq_index, hidden_index]
        return self._array[index]


def _vmlx_wrap_rank3_prompt_cache_for_mlx_vlm(cache):
    """Wrap rank-3 prompt-cache tensors for upstream mlx-vlm trim code."""
    if cache is None:
        return None
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if (
            keys is not None
            and values is not None
            and getattr(keys, "ndim", len(getattr(keys, "shape", ()))) == 3
            and getattr(values, "ndim", len(getattr(values, "shape", ()))) == 3
        ):
            wrapped_keys = _Rank3KVTrimView(keys)
            wrapped_values = _Rank3KVTrimView(values)
            layer_cache.keys = wrapped_keys
            layer_cache.values = wrapped_values
            state = getattr(layer_cache, "state", None)
            if state is not None and len(state) >= 2:
                offset = state[2] if len(state) >= 3 else getattr(layer_cache, "offset", 0)
                layer_cache.state = (wrapped_keys, wrapped_values, offset)
    return cache


def _vmlx_trim_prompt_cache(cache, prefix_len: int):
    """Rank-aware prompt-cache trim used by tests and local callers."""
    if cache is None:
        return None
    prefix_len = max(int(prefix_len or 0), 0)
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            continue
        ndim = getattr(keys, "ndim", len(getattr(keys, "shape", ())))
        # A live KVCache allocates in `step` chunks, so keys.shape[seq] is the
        # BUFFER size and is routinely larger than the token count. Comparing
        # prefix_len against it (and then assigning offset = prefix_len) would
        # raise the offset into allocation slack and turn zero rows into
        # tokens. Clamp through the logical extent instead.
        if ndim >= 4:
            target = logical_truncate_target(layer_cache, prefix_len, keys.shape[2])
            if target < keys.shape[2]:
                layer_cache.keys = keys[:, :, :target, :]
                layer_cache.values = values[:, :, :target, :]
            if hasattr(layer_cache, "offset"):
                layer_cache.offset = target
        elif ndim == 3:
            target = logical_truncate_target(layer_cache, prefix_len, keys.shape[1])
            if target < keys.shape[1]:
                layer_cache.keys = keys[:, :target, :]
                layer_cache.values = values[:, :target, :]
            if hasattr(layer_cache, "offset"):
                layer_cache.offset = target
    return cache


def _vmlx_prime_qwen_mrope_for_full_prompt(model, input_ids, mask, kwargs) -> bool:
    """Prime Qwen/N2 mRoPE state before full multimodal cached prefill.

    Upstream already computes this state before cached-prefix suffix reuse. The
    same state is required when a fresh full image/video prompt is forwarded
    with a newly created prompt cache; otherwise Qwen language code can attempt
    ``cache_offset + None`` during first prefill.
    """
    lm = getattr(model, "language_model", None)
    get_rope_index = getattr(lm, "get_rope_index", None)
    if not callable(get_rope_index):
        return True
    if not (hasattr(lm, "_rope_deltas") or hasattr(lm, "_position_ids")):
        return True
    if kwargs.get("rope_deltas", None) is not None:
        if hasattr(lm, "_rope_deltas") and getattr(lm, "_rope_deltas", None) is None:
            lm._rope_deltas = kwargs["rope_deltas"]
        return True
    if kwargs.get("image_grid_thw", None) is None and kwargs.get("video_grid_thw", None) is None:
        return True
    try:
        position_ids, rope_deltas = get_rope_index(
            input_ids,
            kwargs.get("image_grid_thw", None),
            kwargs.get("video_grid_thw", None),
            mask,
        )
    except Exception as exc:
        _logger.warning("mlx_vlm_compat: could not prime Qwen mRoPE state: %s", exc)
        return False
    if hasattr(lm, "_position_ids"):
        lm._position_ids = position_ids
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = rope_deltas
    kwargs["rope_deltas"] = rope_deltas
    return True


def _patch_prompt_cache_rank3_trim() -> None:
    try:
        import importlib
        import inspect

        generate_mod = importlib.import_module("mlx_vlm.generate")
    except Exception:
        return
    if not hasattr(generate_mod, "_vmlx_trim_prompt_cache"):
        generate_mod._vmlx_trim_prompt_cache = _vmlx_trim_prompt_cache
    if not hasattr(generate_mod, "_vmlx_wrap_rank3_prompt_cache_for_mlx_vlm"):
        generate_mod._vmlx_wrap_rank3_prompt_cache_for_mlx_vlm = (
            _vmlx_wrap_rank3_prompt_cache_for_mlx_vlm
        )
    if not hasattr(generate_mod, "_vmlx_prime_qwen_mrope_for_full_prompt"):
        generate_mod._vmlx_prime_qwen_mrope_for_full_prompt = (
            _vmlx_prime_qwen_mrope_for_full_prompt
        )
    stream_generate = getattr(generate_mod, "stream_generate", None)
    if stream_generate is None or getattr(
        stream_generate, "_vmlx_rank3_cache_trim_patched", False
    ) and getattr(
        stream_generate, "_vmlx_mrope_full_prefill_patched", False
    ):
        return
    try:
        source = inspect.getsource(stream_generate)
    except Exception as exc:
        _logger.debug("mlx_vlm_compat: stream_generate source unavailable: %s", exc)
        return
    source = textwrap.dedent(source)
    start = source.find("# Reuse the saved KV cache (trimmed to prefix length)")
    end = source.find('kwargs["prompt_cache"] = kv_cache', start)
    if start < 0 or end < 0:
        _logger.debug("mlx_vlm_compat: stream_generate trim snippet not found")
        return
    end = source.find("\n", end)
    if end < 0:
        return
    old = source[start : end + 1]
    new = """\
# Reuse the saved KV cache (trimmed to prefix length)
            kv_cache = prompt_cache_state.cache
            _vmlx_trim_prompt_cache(kv_cache, prefix_len)
            kwargs["prompt_cache"] = kv_cache
"""
    patched = source[:start] + new + source[end + 1 :]
    needle = "    total_prompt_tokens = reused_prefix_len + input_ids.size\n"
    if needle not in patched:
        _logger.debug("mlx_vlm_compat: stream_generate full-prefill mRoPE insertion point not found")
        return
    patched = patched.replace(
        needle,
        "    _vmlx_prime_qwen_mrope_for_full_prompt(model, input_ids, mask, kwargs)\n"
        + needle,
        1,
    )
    # Compile into the real module namespace, not a copied globals dict.
    # ``stream_generate`` reads mutable module globals such as
    # ``generation_stream``.  A copied namespace freezes the stream object
    # that happened to exist on the thread which first applied this patch;
    # later model-worker rebinding then updates ``mlx_vlm.generate`` but not
    # the patched function's private globals.  Full-suite ordering exposed
    # that as ``There is no Stream(gpu,N) in current thread`` when the compat
    # patch ran on MainThread before SimpleEngine loaded a VLM on its worker.
    namespace = generate_mod.__dict__
    namespace["_vmlx_trim_prompt_cache"] = _vmlx_trim_prompt_cache
    namespace["_vmlx_prime_qwen_mrope_for_full_prompt"] = (
        _vmlx_prime_qwen_mrope_for_full_prompt
    )
    exec(compile(patched, getattr(generate_mod, "__file__", "<mlx_vlm.generate>"), "exec"), namespace)
    patched_fn = namespace.get("stream_generate")
    if patched_fn is not None:
        patched_fn._vmlx_rank3_cache_trim_patched = True
        patched_fn._vmlx_mrope_full_prefill_patched = True
        generate_mod.stream_generate = patched_fn
        _logger.debug("mlx_vlm_compat: patched stream_generate VLM cache compatibility")


def _patch_qwen35_language_mrope_none_delta() -> None:
    try:
        import importlib
        import inspect
    except Exception:
        return

    module_names = (
        "mlx_vlm.models.qwen3_5.language",
        "mlx_vlm.models.qwen3_5_moe.language",
    )
    for module_name in module_names:
        try:
            language_mod = importlib.import_module(module_name)
        except Exception:
            continue
        LanguageModel = getattr(language_mod, "LanguageModel", None)
        if LanguageModel is None:
            continue
        original = getattr(LanguageModel, "__call__", None)
        if original is None or getattr(original, "_vmlx_mrope_none_delta_patched", False):
            continue
        try:
            source = textwrap.dedent(inspect.getsource(original))
        except Exception as exc:
            _logger.debug("mlx_vlm_compat: Qwen language source unavailable: %s", exc)
            continue
        old = """\
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
"""
        new = """\
                rope_deltas = (
                    rope_deltas_kw
                    if rope_deltas_kw is not None
                    else self._rope_deltas
                )
                if cache_offset is None:
                    cache_offset = 0
                if rope_deltas is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
                delta = mx.array(
                    cache_offset + rope_deltas if cache is not None else 0
                )
"""
        if old not in source:
            _logger.debug("mlx_vlm_compat: Qwen language delta snippet not found in %s", module_name)
            continue
        patched = source.replace(old, new, 1)
        offset_old = """\
                cache_offsets = mx.maximum(c0.offset, 0)

        # Check if mask shape matches input shape (for chunked prefill compatibility)
"""
        offset_new = """\
                cache_offsets = mx.maximum(c0.offset, 0)
            if cache_offset is None:
                cache_offset = 0
            if isinstance(cache_offset, mx.array) and cache_offset.size == 1:
                cache_offset = int(cache_offset.item())

        # Check if mask shape matches input shape (for chunked prefill compatibility)
"""
        if offset_old in patched:
            patched = patched.replace(offset_old, offset_new, 1)
        namespace = dict(language_mod.__dict__)
        try:
            exec(
                compile(
                    patched,
                    getattr(language_mod, "__file__", "<qwen_language>"),
                    "exec",
                ),
                namespace,
            )
        except Exception as exc:
            _logger.debug("mlx_vlm_compat: Qwen language patch compile failed: %s", exc)
            continue
        patched_fn = namespace.get("__call__")
        if patched_fn is None:
            continue
        patched_fn._vmlx_mrope_none_delta_patched = True
        LanguageModel.__call__ = patched_fn
    _logger.debug("mlx_vlm_compat: patched Qwen3.5/N2 language mRoPE delta fallback")


def _patch_qwen3_vl_grid_thw() -> None:
    try:
        import mlx.core as mx
        from mlx_vlm.models.qwen3_vl import vision as _qv
    except ImportError:
        return

    VisionModel = getattr(_qv, "VisionModel", None)
    if VisionModel is None:
        return

    def _as_mx(x):
        if isinstance(x, mx.array):
            return x
        try:
            return mx.array(x)
        except Exception:
            return x

    orig_rot = VisionModel.rot_pos_emb
    if not getattr(orig_rot, "_vmlx_patched", False):
        def rot_pos_emb(self, grid_thw):
            return orig_rot(self, _as_mx(grid_thw))
        rot_pos_emb._vmlx_patched = True  # type: ignore[attr-defined]
        VisionModel.rot_pos_emb = rot_pos_emb  # type: ignore[assignment]

    orig_call = VisionModel.__call__
    if not getattr(orig_call, "_vmlx_patched", False):
        def __call__(self, hidden_states, grid_thw, **kwargs):
            return orig_call(self, hidden_states, _as_mx(grid_thw), **kwargs)
        __call__._vmlx_patched = True  # type: ignore[attr-defined]
        VisionModel.__call__ = __call__  # type: ignore[assignment]

    _logger.debug("mlx_vlm_compat: patched Qwen3-VL VisionModel grid_thw coercion")


def _qwen35_patch_embed_to_mlx_layout(key, value):
    if (
        str(key).endswith("patch_embed.proj.weight")
        and getattr(value, "ndim", None) == 5
        and int(value.shape[1]) in (1, 3)
        and int(value.shape[-1]) not in (1, 3)
    ):
        return value.transpose(0, 2, 3, 4, 1)
    return value


def _patch_lfm2_vl_input_embeddings() -> None:
    """LFM2-VL (mlx-vlm 0.5.0) cannot serve a single image request.

    Its own ``Model.__call__`` invokes
    ``self.get_input_embeddings(input_ids, pixel_values, spatial_shapes,
    pixel_attention_mask)`` — four positional arguments — while the method
    is declared ``(self, input_ids=None, pixel_values=None, **kwargs)`` and
    reads the extras out of ``kwargs``. Every image prompt therefore dies
    with "get_input_embeddings() takes from 1 to 3 positional arguments but
    5 were given" before any Metal work happens; text-only prompts are
    unaffected because that path never passes the extras.

    Accept the extra positionals and forward them as the keywords the body
    already looks for. Callers that pass keywords are unaffected.
    """
    try:
        from mlx_vlm.models.lfm2_vl import lfm2_vl as _lfm2_vl
    except ImportError:
        return

    Model = getattr(_lfm2_vl, "Model", None)
    if Model is None:
        return
    original = getattr(Model, "get_input_embeddings", None)
    if original is None or getattr(original, "_vmlx_lfm2_vl_positional", False):
        return

    def get_input_embeddings(
        self,
        input_ids=None,
        pixel_values=None,
        spatial_shapes=None,
        pixel_attention_mask=None,
        _original=original,
        **kwargs,
    ):
        if spatial_shapes is not None:
            kwargs.setdefault("spatial_shapes", spatial_shapes)
        if pixel_attention_mask is not None:
            kwargs.setdefault("pixel_attention_mask", pixel_attention_mask)
        return _original(self, input_ids, pixel_values, **kwargs)

    get_input_embeddings._vmlx_lfm2_vl_positional = True  # type: ignore[attr-defined]
    Model.get_input_embeddings = get_input_embeddings  # type: ignore[assignment]

    # Second defect in the same module: Model.__call__ forwards the whole
    # InputEmbeddingsFeatures dataclass as `inputs_embeds`, but
    # LanguageModel.__call__ is typed Optional[mx.array] and indexes it —
    # "'InputEmbeddingsFeatures' object has no attribute 'shape'". Unwrap at
    # the language-model boundary so it is fixed no matter which caller
    # produced the features object.
    try:
        from mlx_vlm.models.lfm2_vl import language as _lfm2_lang
    except ImportError:
        _lfm2_lang = None
    LanguageModel = getattr(_lfm2_lang, "LanguageModel", None) if _lfm2_lang else None
    if LanguageModel is not None:
        lm_original = getattr(LanguageModel, "__call__", None)
        if lm_original is not None and not getattr(
            lm_original, "_vmlx_lfm2_vl_unwrap", False
        ):

            def lm_call(
                self,
                inputs,
                mask=None,
                cache=None,
                inputs_embeds=None,
                _original=lm_original,
                **kwargs,
            ):
                if inputs_embeds is not None and not hasattr(inputs_embeds, "shape"):
                    inputs_embeds = getattr(
                        inputs_embeds, "inputs_embeds", inputs_embeds
                    )
                return _original(self, inputs, mask, cache, inputs_embeds, **kwargs)

            lm_call._vmlx_lfm2_vl_unwrap = True  # type: ignore[attr-defined]
            LanguageModel.__call__ = lm_call  # type: ignore[assignment]

    _logger.info(
        "mlx_vlm_compat: LFM2-VL image path patched "
        "(get_input_embeddings arity + inputs_embeds unwrap)"
    )


def _patch_qwen35_patch_embed_layout() -> None:
    try:
        from mlx_vlm.models.qwen3_5 import qwen3_5 as _qwen_vl
    except ImportError:
        _qwen_vl = None
    try:
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as _qwen_moe_vl
    except ImportError:
        _qwen_moe_vl = None

    for module in (_qwen_vl, _qwen_moe_vl):
        Model = getattr(module, "Model", None)
        if Model is None:
            continue
        original = getattr(Model, "sanitize", None)
        if original is None or getattr(original, "_vmlx_patch_embed_layout", False):
            continue

        def sanitize(self, weights, _original=original):
            fixed = {}
            for key, value in weights.items():
                fixed[key] = _qwen35_patch_embed_to_mlx_layout(key, value)
            return _original(self, fixed)

        sanitize._vmlx_patch_embed_layout = True  # type: ignore[attr-defined]
        Model.sanitize = sanitize  # type: ignore[assignment]

    _logger.debug("mlx_vlm_compat: patched Qwen3.5/3.6 patch_embed layout")


def _patch_qwen3_vl_vision_model_type_allowlist() -> None:
    """Allow additional Qwen3.5/3.6 VL vision `model_type` strings.

    Upstream mlx_vlm/models/qwen3_vl/vision.py VisionModel.__init__ hard-codes
    a check `if self.model_type not in ["qwen3_vl", "qwen3_5", "qwen3_5_moe"]`
    and raises `ValueError: Unsupported model type: qwen3_5_moe_vision`. But
    JANG-authored Ornith bundles stamp `vision_config.model_type` as
    `qwen3_5_moe_vision` (the granular vision-tower type, distinct from the
    text-side type), which trips the check and blocks every VL load path —
    including Smelt (which routes through `jang_tools.loader._load_jang_v2_vlm`
    even for text-only sessions, before force_text_only routing kicks in).

    Patch: wrap VisionModel.__init__ so `qwen3_5_moe_vision` (and any future
    `*_vision` sibling) are treated as `qwen3_5_moe` at model-type-check time,
    while preserving the original `self.model_type` string for anything
    downstream that inspects it.

    Idempotent. No-op if mlx_vlm not installed or the check has been changed.
    """
    try:
        import importlib
        vision_mod = importlib.import_module("mlx_vlm.models.qwen3_vl.vision")
    except Exception as exc:
        _logger.debug("mlx_vlm_compat: qwen3_vl vision module unavailable: %s", exc)
        return

    VisionModel = getattr(vision_mod, "VisionModel", None)
    if VisionModel is None:
        return
    original_init = getattr(VisionModel, "__init__", None)
    if original_init is None or getattr(original_init, "_vmlx_vision_allowlist_patched", False):
        return

    # Broader set: text-side + explicit vision-side variants seen in JANG bundles.
    _ALLOWED = {"qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_vision", "qwen3_5_vision"}

    def patched_init(self, config, _original=original_init):
        # If the config's model_type is a *_vision sibling of an allowed root,
        # temporarily normalize so the upstream check passes, then restore.
        raw = getattr(config, "model_type", None)
        stashed = None
        try:
            if isinstance(raw, str) and raw in _ALLOWED and raw not in ("qwen3_vl", "qwen3_5", "qwen3_5_moe"):
                # rewrite to the closest supported root
                normalized = "qwen3_5_moe" if "moe" in raw else "qwen3_5"
                stashed = raw
                config.model_type = normalized
            _original(self, config)
        finally:
            if stashed is not None:
                # restore original model_type on both the config and self so
                # downstream code that keys on the exact string still works.
                config.model_type = stashed
                self.model_type = stashed

    patched_init._vmlx_vision_allowlist_patched = True  # type: ignore[attr-defined]
    VisionModel.__init__ = patched_init  # type: ignore[assignment]
    _logger.debug(
        "mlx_vlm_compat: patched qwen3_vl VisionModel allowlist (adds qwen3_5_moe_vision, qwen3_5_vision)"
    )


def _patch_qwen35_mixed_image_video_embeddings() -> None:
    """Encode BOTH image and video pixels when a conversation carries both.

    Upstream ``qwen3_5.Model.get_input_embeddings`` reads
    ``pixel_values_videos`` only as a FALLBACK (``if pixel_values is None``)
    and picks ``grid_thw = image_grid_thw or video_grid_thw``. With an image
    and a video in the same conversation, the vision tower therefore encodes
    ONLY the image, and the sequential merge scatters those image features
    into the VIDEO token positions as well.

    Measured live on Qwen3.6-27B-4D: the processor emits byte-identical video
    tensors with or without an image present (probe: identical
    pixel_values_videos, grid [2,32,32]), yet the model in the mixed case
    enumerates ~12 identical frames pinned at fixed coordinates — the image,
    replicated where the ball should be. Video-only turns answer "moves
    right" because the fallback fires.

    Fix: encode image and video pixel sets separately with their own grids
    and concatenate the feature rows in the order their pad tokens appear in
    ``input_ids`` (the merge fills special positions sequentially). The
    vision feature cache is only consulted for the image-only case, where its
    key covers all encoded content.
    """
    try:
        from mlx_vlm.models.qwen3_5 import qwen3_5 as _qvl
    except Exception as exc:  # pragma: no cover
        _logger.debug("mixed image+video patch skipped: %s", exc)
        return
    cls = _qvl.Model
    if getattr(cls.get_input_embeddings, "_vmlx_mixed_media_patched", False):
        return

    _original = cls.get_input_embeddings

    def patched_get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        # The generator historically shipped the video tensor as
        # `video_pixel_values` while every model reader (upstream and vendor)
        # spells it `pixel_values_videos`. Accept both here so neither caller
        # generation strands the tensor in an ignored kwarg.
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        if pixel_values_videos is None:
            pixel_values_videos = kwargs.get("video_pixel_values", None)
        import os as _dbg_os
        if _dbg_os.environ.get("VMLX_DEBUG_MEDIA_KWARGS") == "1":
            _logger.info(
                "media kwargs: pixel_values=%s pixel_values_videos=%s keys=%s",
                None if pixel_values is None else tuple(pixel_values.shape),
                None if pixel_values_videos is None else tuple(pixel_values_videos.shape),
                sorted(k for k in kwargs if "pixel" in k or "grid" in k),
            )
        if pixel_values is None and pixel_values_videos is not None:
            # Video-only: give the tensor to upstream under the POSITIONAL
            # slot it actually consumes (the vendored runtime has no
            # pixel_values_videos fallback at all).
            return _original(
                self, input_ids=input_ids, pixel_values=pixel_values_videos, **kwargs
            )
        if pixel_values is None or pixel_values_videos is None:
            # Single-media (or text-only): upstream handles these correctly.
            return _original(self, input_ids=input_ids, pixel_values=pixel_values, **kwargs)

        import mlx.core as mx

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        image_feats, _ = self.vision_tower(pixel_values.astype(dtype), image_grid_thw)
        video_feats, _ = self.vision_tower(
            pixel_values_videos.astype(dtype), video_grid_thw
        )

        # The merge fills image/video token positions sequentially from one
        # feature matrix, so the concatenation order must match the order the
        # pad tokens appear in the prompt.
        ids = input_ids[0] if input_ids.ndim > 1 else input_ids
        img_tok = int(self.config.image_token_index)
        vid_tok = int(self.config.video_token_index)
        id_list = ids.tolist()
        img_first = (
            id_list.index(img_tok) if img_tok in id_list else len(id_list)
        ) <= (id_list.index(vid_tok) if vid_tok in id_list else len(id_list))
        hidden_states = (
            mx.concatenate([image_feats, video_feats], axis=0)
            if img_first
            else mx.concatenate([video_feats, image_feats], axis=0)
        )

        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )
        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = rope_deltas
        return _qvl.InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    patched_get_input_embeddings._vmlx_mixed_media_patched = True  # type: ignore[attr-defined]
    cls.get_input_embeddings = patched_get_input_embeddings  # type: ignore[assignment]
    _logger.info("mlx_vlm_compat: qwen3_5 mixed image+video embeddings patched")
