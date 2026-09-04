"""The MTP priming sidecar must be keyed at a boundary the backbone prefix
cache can actually restore: never past the prompt's N-1 terminal boundary,
even when the prompt length is an exact multiple of the cache block."""

import mlx.core as mx

from vmlx_engine.native_mtp_prompt_priming import _PrimeContext, _capture_boundary


def _ctx(prompt_len, block=64):
    ctx = _PrimeContext(mtp_cache=[], prompt_tokens=tuple(range(prompt_len)),
                        block_size=block, prefix_cache=object())
    ctx.folded = prompt_len  # everything folded already
    return ctx


def test_block_aligned_prompt_keys_sidecar_at_terminal_boundary():
    # 5952 = 93 x 64: the full-block boundary equals the prompt length, but
    # the prefix cache restores at 5951 -> the sidecar must sit at 5951.
    ctx = _ctx(5952)
    hidden = mx.zeros((1, 5952, 8))
    _capture_boundary(ctx, hidden, 0, 5952)
    assert ctx.boundary_candidate is not None
    assert ctx.boundary_candidate.boundary_tokens == 5951


def test_unaligned_prompt_keys_sidecar_at_terminal_boundary():
    ctx = _ctx(4754)
    hidden = mx.zeros((1, 4754, 8))
    _capture_boundary(ctx, hidden, 0, 4754)
    assert ctx.boundary_candidate.boundary_tokens == 4753


def test_intermediate_chunk_keeps_full_block_boundary():
    # A chunk that ends before the terminal keeps its block-aligned key.
    ctx = _ctx(5000)
    hidden = mx.zeros((1, 2048, 8))
    _capture_boundary(ctx, hidden, 0, 2048)
    assert ctx.boundary_candidate.boundary_tokens == 2048
