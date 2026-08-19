"""DFlash2 bridge for text-only Qwen VLM generation.

The draft implementation comes from z-lab/dflash. This bridge adapts vMLX's
Qwen VLM language wrapper and hybrid rollback API to its MLX generation loop.
"""

from __future__ import annotations

from typing import Any, Iterator


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

    import dflash.model_mlx as runtime

    adapter = _adapter_for(model)
    original_capture = runtime._GDNStateCapture
    runtime._GDNStateCapture = lambda: _VLMGDNStateCapture(adapter)
    try:
        yield from runtime.stream_generate(
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
    finally:
        runtime._GDNStateCapture = original_capture
