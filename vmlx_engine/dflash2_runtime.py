"""DFlash2 bridge for text-only Qwen VLM generation.

The draft implementation comes from z-lab/dflash. This bridge adapts vMLX's
Qwen VLM language wrapper and hybrid rollback API to its MLX generation loop.

Multiturn prefix reuse: the DFlash2 lane runs through SimpleEngine, which has
no paged prefix-cache stack, so upstream's loop re-prefills the entire
conversation every turn (measured 15.5 s/turn at 7k tokens). This module keeps
an in-process session store of (confirmed tokens, target cache, draft cache)
per finished turn and, when a new prompt extends a stored conversation,
prefills only the delta. The generation loop is adapted from the pinned
``dflash==0.1.0`` runtime (``dflash/model_mlx.py``); its cache invariant is
that the last emitted token is sampled but never forwarded, so a stored cache
always holds exactly ``confirmed[:-1]`` positions.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


def _prefix_reuse_enabled() -> bool:
    return os.environ.get("VMLX_DFLASH2_PREFIX_REUSE", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


class _TargetAdapter:
    def __init__(self, language_model: Any):
        self._target = language_model
        self.model = language_model.model
        self.gdn_states: list[Any] = []

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, inputs, cache=None):
        hidden = self.model(
            inputs,
            cache=cache,
            gdn_sink=self.gdn_states,
            return_unnormed=True,
        )
        hidden = self.model.norm(hidden)
        return self._target.lm_head(hidden)

    def __getattr__(self, name: str):
        return getattr(self._target, name)


class _VLMGDNStateCapture:
    def __init__(self, adapter: _TargetAdapter):
        self.adapter = adapter

    def clear(self) -> None:
        self.adapter.gdn_states.clear()

    def close(self) -> None:
        return None

    def rollback(self, cache, accepted, trim) -> None:
        block_size = int(trim) + int(accepted) + 1
        rollback = getattr(self.adapter._target, "rollback_speculative_cache", None)
        if rollback is not None:
            rollback(cache, self.adapter.gdn_states, int(accepted), block_size)
            return

        import mlx.core as mx
        from mlx_lm.models.gated_delta import gated_delta_update

        gdn_index = 0
        for layer_cache in cache:
            if layer_cache.is_trimmable():
                layer_cache.trim(int(trim))
                continue
            (
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                initial_state,
                mask,
                conv_input,
                kernel_size,
            ) = self.adapter.gdn_states[gdn_index]
            count = int(accepted) + 1
            _, state = gated_delta_update(
                q[:, :count],
                k[:, :count],
                v[:, :count],
                a[:, :count],
                b[:, :count],
                a_log,
                dt_bias,
                initial_state,
                None if mask is None else mask[:, :count],
                use_kernel=True,
            )
            layer_cache[1] = state
            layer_cache[0] = mx.contiguous(
                conv_input[:, count : count + int(kernel_size) - 1]
            )
            gdn_index += 1


def _adapter_for(model: Any) -> _TargetAdapter:
    language_model = model.language_model
    adapter = getattr(language_model, "_vmlx_dflash2_adapter", None)
    if adapter is None:
        language_model._vmlx_force_text_rope_1d = True
        adapter = _TargetAdapter(language_model)
        language_model._vmlx_dflash2_adapter = adapter
    return adapter


# ---------------------------------------------------------------------------
# Session store for multiturn prefix reuse
# ---------------------------------------------------------------------------


class _DFlash2SessionStore:
    """LRU store of finished-turn generation state, newest last.

    Each entry: ``tokens`` (full confirmed token list; the target cache holds
    exactly ``tokens[:-1]`` forwarded positions), ``target_cache``,
    ``draft_cache`` (or None when its offset could not be aligned), and
    ``model_key`` guarding against cross-model token-id collisions.
    """

    def __init__(self, max_entries: int = 2):
        self.max_entries = int(max_entries)
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def take_matching(self, model_key: Any, prompt_tokens: list[int]) -> Optional[dict]:
        """Pop and return the longest stored conversation that is a prefix of
        ``prompt_tokens``. Ownership transfers to the caller."""
        with self._lock:
            best_i = -1
            best_len = 0
            for i, entry in enumerate(self._entries):
                if entry["model_key"] != model_key:
                    continue
                stored = entry["tokens"]
                if len(stored) > len(prompt_tokens):
                    continue
                if prompt_tokens[: len(stored)] == stored and len(stored) > best_len:
                    best_i = i
                    best_len = len(stored)
            if best_i < 0:
                return None
            return self._entries.pop(best_i)

    def put(self, entry: dict) -> None:
        with self._lock:
            self._entries.append(entry)
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_SESSION_STORE = _DFlash2SessionStore()


def _checkpoint_correction(cycle_committed: int, cycle_kept: int):
    """Rollback args to shrink the final cycle's committed positions down to
    the emitted ones. Returns (accepted, trim) for capture.rollback / the
    incremental trim count for trimmable caches, or None when already exact.

    ``cycle_committed`` counts target-cache positions the final cycle still
    holds (bs when it exited before its trim, accepted+1 after a normal trim);
    ``cycle_kept`` counts tokens actually emitted from that cycle. The cache
    must end at confirmed[:-1], and the cycle's first committed position is
    the previous turn token, so it must keep exactly ``cycle_kept`` positions.
    """
    excess = int(cycle_committed) - int(cycle_kept)
    if excess <= 0:
        return None
    # accepted+1 positions survive a rollback of block_size=cycle_committed.
    return (int(cycle_kept) - 1, excess)


def _stream_generate_resumable(
    model: Any,
    adapter: _TargetAdapter,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    *,
    block_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    prefill_step_size: int = 2048,
) -> Iterator[Any]:
    """The dflash==0.1.0 ``_stream_generate`` loop with session resume.

    Differences from upstream: (1) when the session store holds a conversation
    that prefixes this prompt, only the delta is prefilled into its caches;
    (2) on clean EOS/length exit the caches are rolled back to exactly
    ``confirmed[:-1]`` and stored for the next turn. Everything else is kept
    line-for-line so behaviour off the resume path is identical.
    """
    import mlx.core as mx
    import dflash.model_mlx as runtime

    runtime._patch_model(adapter, draft.config.target_layer_ids)
    sampler = runtime.make_sampler(temperature, top_p, top_k)

    if not isinstance(tokenizer, runtime.TokenizerWrapper):
        tokenizer = runtime.TokenizerWrapper(tokenizer)

    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    prompt_list = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=add_special_tokens)]
    prompt_arr = mx.array(prompt_list)

    detokenizer = tokenizer.detokenizer
    mask_id = int(draft.config.mask_token_id)
    tokens: list[int] = []

    model_key = (id(model), id(draft))
    resume = (
        _SESSION_STORE.take_matching(model_key, prompt_list)
        if _prefix_reuse_enabled()
        else None
    )

    if resume is not None:
        target_cache = resume["target_cache"]
        cache_len = len(resume["tokens"]) - 1
        delta = prompt_arr[cache_len:]
        stored_draft_cache = resume["draft_cache"]
    else:
        target_cache = runtime.make_prompt_cache(adapter)
        cache_len = 0
        delta = prompt_arr
        stored_draft_cache = None

    draft.bind(adapter)
    _target_can_trim = runtime.can_trim_prompt_cache(target_cache)
    _capture = None

    try:
        tic = time.perf_counter()
        with mx.stream(runtime.generation_stream):
            hidden_limit = (
                draft.config.sliding_window - 1
                if all(t == "sliding_attention" for t in draft.config.layer_types)
                else None
            )
            logits, hidden, hidden_offset = runtime._prefill_target(
                adapter, delta, target_cache, hidden_limit, prefill_step_size
            )
            if stored_draft_cache is not None and hidden.shape[1] == delta.size:
                # Stored draft KV ends exactly where this hidden window
                # starts, so the drafter keeps its history.
                draft_cache = stored_draft_cache
            else:
                draft_cache = runtime.make_prompt_cache(draft)
                for cache in draft_cache:
                    cache.offset = cache_len + int(hidden_offset)
        mx.eval(logits, hidden)
        prefill_elapsed = time.perf_counter() - tic
        prompt_tps = prompt_arr.size / max(prefill_elapsed, 1e-9)
        if resume is not None:
            logger.info(
                "DFlash2 prefix reuse: %d of %d prompt tokens from session store "
                "(prefilled %d, draft cache %s, %.2fs)",
                cache_len,
                len(prompt_list),
                int(delta.size),
                "spliced" if draft_cache is stored_draft_cache else "rebuilt",
                prefill_elapsed,
            )

        tic = time.perf_counter()
        token = sampler(logits[:, -1:])[0, 0].item()
        tokens.append(token)
        n = 1

        # Final-cycle bookkeeping for the exit checkpoint. Before any verify
        # cycle runs, the cache holds exactly the prompt = confirmed[:-1], so
        # committed == kept.
        cycle_committed = 0
        cycle_kept = 0

        def _checkpoint() -> None:
            if not _prefix_reuse_enabled():
                return
            correction = _checkpoint_correction(cycle_committed, cycle_kept)
            if correction is not None:
                accepted_i, trim_i = correction
                if _target_can_trim:
                    runtime._trim_recent_cache(target_cache, trim_i)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted_i, trim_i)
                else:
                    return  # cannot align this cache; skip storing
            confirmed = prompt_list + [int(t) for t in tokens]
            draft_trim = draft_cache[0].offset - (len(confirmed) - 1)
            store_draft: Optional[list] = draft_cache
            if draft_trim > 0:
                runtime._trim_recent_cache(draft_cache, int(draft_trim))
            if draft_cache[0].offset != len(confirmed) - 1:
                store_draft = None
            _SESSION_STORE.put(
                {
                    "model_key": model_key,
                    "tokens": confirmed,
                    "target_cache": target_cache,
                    "draft_cache": store_draft,
                }
            )
            logger.info(
                "DFlash2 session stored: %d confirmed tokens (draft cache %s)",
                len(confirmed),
                "kept" if store_draft is not None else "dropped",
            )

        if token in tokenizer.eos_token_ids:
            detokenizer.add_token(token)
            detokenizer.finalize()
            _checkpoint()
            yield runtime._make_response(
                detokenizer.last_segment, [token], None, prompt_arr.size, prompt_tps, n, tic, "stop"
            )
            return

        detokenizer.add_token(token)
        yield runtime._make_response(
            detokenizer.last_segment,
            [token],
            None,
            prompt_arr.size,
            prompt_tps,
            n,
            tic,
        )

        if not _target_can_trim:
            _capture = _VLMGDNStateCapture(adapter)

        while n < max_tokens:
            bs = min(block_size, max_tokens - n + 1)
            if bs <= 1:
                break

            with mx.stream(runtime.generation_stream):
                block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
                if isinstance(draft, runtime.DFlash2DraftModel):
                    draft_tokens, draft_indices, draft_probs = draft.propose(
                        block, hidden, draft_cache, temperature, logits_start=1
                    )
                else:
                    draft_logits = draft(block, hidden, draft_cache, logits_start=1)
                    if temperature > 0:
                        draft_probs = runtime._sampling_probs(
                            draft_logits, temperature, top_p, top_k
                        )
                        draft_tokens = runtime._sample_probs(draft_probs)
                        draft_indices = None
                    else:
                        draft_tokens = mx.argmax(draft_logits, axis=-1)
                if (trim_n := draft_cache[0].offset - (prompt_arr.size + n - 1)) > 0:
                    runtime._trim_recent_cache(draft_cache, trim_n)
            mx.async_eval(draft_tokens)

            if _capture is not None:
                _capture.clear()
            with mx.stream(runtime.generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[tokens[-1]]]), draft_tokens], axis=1
                )
                logits = adapter(verify_input, target_cache)
                hidden = mx.concatenate(adapter._hidden_states, axis=-1)
                if temperature > 0:
                    target_probs = runtime._sampling_probs(
                        logits, temperature, top_p, top_k
                    )
                else:
                    target_tokens = mx.argmax(logits, axis=-1)
            mx.async_eval(target_probs if temperature > 0 else target_tokens, hidden)

            d_list = draft_tokens[0].tolist()
            if temperature > 0:
                accepted, bonus = runtime._rejection_sample(
                    draft_tokens, target_probs, draft_probs, draft_indices
                )
            else:
                t_list = target_tokens[0].tolist()
                accepted = next(
                    (i for i in range(len(d_list)) if d_list[i] != t_list[i]),
                    len(d_list),
                )
                bonus = t_list[accepted]
            new_tokens = d_list[:accepted] + [bonus]
            new_tokens = new_tokens[: max_tokens - n]

            eos_idx = next(
                (i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids),
                None,
            )
            if eos_idx is not None:
                new_tokens = new_tokens[: eos_idx + 1]
                for t in new_tokens:
                    detokenizer.add_token(t)
                detokenizer.finalize()
                tokens.extend(new_tokens)
                n += len(new_tokens)
                # The upstream loop returns before its rollback, leaving all
                # bs verify positions committed.
                cycle_committed = bs
                cycle_kept = len(new_tokens)
                _checkpoint()
                yield runtime._make_response(
                    detokenizer.last_segment,
                    new_tokens,
                    len(new_tokens),
                    prompt_arr.size,
                    prompt_tps,
                    n,
                    tic,
                    "stop",
                )
                return

            for t in new_tokens:
                detokenizer.add_token(t)
            tokens.extend(new_tokens)
            previous_n = n
            n += len(new_tokens)

            if n // 256 > previous_n // 256:
                mx.clear_cache()

            yield runtime._make_response(
                detokenizer.last_segment,
                new_tokens,
                len(new_tokens),
                prompt_arr.size,
                prompt_tps,
                n,
                tic,
            )

            trim = bs - accepted - 1
            if trim > 0:
                if _target_can_trim:
                    runtime._trim_recent_cache(target_cache, trim)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted, trim)
            hidden = hidden[:, : accepted + 1, :]
            cycle_committed = accepted + 1
            cycle_kept = len(new_tokens)

        detokenizer.finalize()
        _checkpoint()
        yield runtime._make_response(
            detokenizer.last_segment,
            [],
            None,
            prompt_arr.size,
            prompt_tps,
            n,
            tic,
            "length",
        )
    finally:
        if _capture is not None:
            _capture.close()


def stream_dflash2_generate(
    model: Any,
    tokenizer: Any,
    draft: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Iterator[Any]:
    """Yield upstream DFlash2 chunks using vMLX's hybrid Qwen target."""

    import mlx.core as mx
    import dflash.model_mlx as runtime

    adapter = _adapter_for(model)
    with runtime.wired_limit(adapter, [runtime.generation_stream]):
        yield from _stream_generate_resumable(
            model,
            adapter,
            draft,
            tokenizer,
            prompt,
            # Qwen3.8's published/runtime-validated DFlash2 lane is block 5.
            # The checkpoint's training maximum is larger, but using it as
            # the serving block makes four-row verification become seq8 and
            # cuts throughput roughly in half on M5 Max.
            block_size=min(5, int(draft.config.block_size)),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )
