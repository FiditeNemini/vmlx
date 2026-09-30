"""Lossless GLM sequence blocks plus a separate recurrent checkpoint.

This codec does not admit cache hits or publish files. Its caller must bind
every block to model, media and causal token identity, and publish the exact
KDA checkpoint only after all referenced blocks are durable.
"""
from dataclasses import dataclass

import mlx.core as mx

from ..models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache


@dataclass(frozen=True)
class NativeSequenceBlock:
    start: int
    end: int
    # None denotes a recurrent layer, never a missing attention layer.
    layers: tuple


@dataclass(frozen=True)
class NativeBoundary:
    offset: int
    metadata: tuple
    recurrent: tuple
    # Pool materialization is independent of logical token length.
    pooled: tuple


def _axes(meta):
    return (2, 1, 1, None) if meta[1] == "mla_absorbed" else (2, 2, 1, 1)


def _slice(value, axis, start, end):
    selection = [slice(None)] * value.ndim
    selection[axis] = slice(start, end)
    return value[tuple(selection)]


def split_native_sequence(layers, block_size):
    """Split realized native state without synthesizing intermediate KDA.

    Full blocks are immutable sequence fragments. The final partial block
    remains distinct, so extension never overwrites an earlier boundary.
    Block sizes must align with every DSA pool factor.
    """
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid native block size")
    if not layers or {type(x) for x in layers} != {Glm5KDACache, Glm5MLACache}:
        raise ValueError("expected complete GLM native layout")
    metadata, recurrent, pooled, states = [], [], [], []
    offset = None
    for layer in layers:
        meta = tuple(layer.meta_state)
        state = tuple(layer.state)
        type(layer).from_state(state, meta)
        metadata.append(meta)
        states.append(state)
        if isinstance(layer, Glm5KDACache):
            if any(x is None for x in state):
                raise ValueError("incomplete recurrent checkpoint")
            recurrent.append(state)
            pooled.append(None)
        else:
            if block_size % layer.kpool:
                raise ValueError("native block size must align with DSA pools")
            if offset is not None and offset != layer.offset:
                raise ValueError("native layer boundaries disagree")
            offset = layer.offset
            recurrent.append(None)
            pooled.append(bool(state[2 if layer.absorbed else 3].size))
    if not offset:
        raise ValueError("empty native boundary")
    blocks = []
    for start in range(0, offset, block_size):
        end = min(start + block_size, offset)
        fragments = []
        for meta, state, has_pool in zip(metadata, states, pooled):
            if has_pool is None:
                fragments.append(None)
                continue
            pool_slot = 2 if meta[1] == "mla_absorbed" else 3
            arrays = []
            for slot, (array, axis) in enumerate(zip(state, _axes(meta))):
                if axis is None or (slot == pool_slot and not has_pool):
                    arrays.append(array)
                elif slot == pool_slot:
                    arrays.append(_slice(array, axis, start // int(meta[2]), end // int(meta[2])))
                else:
                    arrays.append(_slice(array, axis, start, end))
            fragments.append(tuple(arrays))
        blocks.append(NativeSequenceBlock(start, end, tuple(fragments)))
    return NativeBoundary(offset, tuple(metadata), tuple(recurrent), tuple(pooled)), blocks


def restore_native_sequence(boundary, blocks):
    """Validate contiguous complete fragments, then rebuild typed state."""
    count = len(boundary.metadata)
    if not count or len(boundary.recurrent) != count or len(boundary.pooled) != count:
        raise ValueError("incomplete native boundary descriptor")
    cursor = 0
    for block in blocks:
        if block.start != cursor or block.end <= cursor or len(block.layers) != count:
            raise ValueError("missing, reordered or malformed native block")
        cursor = block.end
    if cursor != boundary.offset:
        raise ValueError("native blocks do not cover checkpoint")
    restored = []
    for index, (meta, recurrent, pooled) in enumerate(zip(
        boundary.metadata, boundary.recurrent, boundary.pooled
    )):
        if pooled is None:
            if recurrent is None or any(b.layers[index] is not None for b in blocks):
                raise ValueError("invalid recurrent checkpoint placement")
            restored.append(Glm5KDACache.from_state(recurrent, meta))
            continue
        if type(pooled) is not bool or recurrent is not None:
            raise ValueError("invalid native attention descriptor")
        pool_slot = 2 if meta[1] == "mla_absorbed" else 3
        arrays = []
        for slot, axis in enumerate(_axes(meta)):
            parts = []
            for block in blocks:
                fragment = block.layers[index]
                if fragment is None or len(fragment) != 4:
                    raise ValueError("missing native attention fragment")
                array = fragment[slot]
                if axis is None:
                    if array.size:
                        raise ValueError("nonempty reserved native state")
                else:
                    expected = block.end - block.start
                    if slot == pool_slot:
                        expected = (block.end // int(meta[2]) - block.start // int(meta[2])) if pooled else 0
                    if array.ndim <= axis or array.shape[axis] != expected:
                        raise ValueError("native fragment length mismatch")
                if parts and (array.dtype != parts[0].dtype or
                    tuple(s for a, s in enumerate(array.shape) if a != axis) !=
                    tuple(s for a, s in enumerate(parts[0].shape) if a != axis)):
                    raise ValueError("native fragment dtype or shape mismatch")
                parts.append(array)
            arrays.append(parts[0] if axis is None else mx.concatenate(parts, axis=axis))
        restored.append(Glm5MLACache.from_state(arrays, meta))
    return restored
