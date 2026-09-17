"""Bind packed PLE rows after graph construction; keep GPU dequantization lazy.

Uses strict zero-copy owned host buffers. Admits all-shard-uniform affine metadata and exact
B1/S1 disjoint per-head hashing only. No fallback/replay after graph build.
"""
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .deferred_ple import _active, Declined


class PackedOwnedLeaf:
    """Immutable-after-fill backing retained by MLX's strict shared import.

    Source-audited preflight is MANDATORY: Python cannot intercept an early
    graph evaluation. No reference to this graph may escape before flush.
    """
    def __init__(self, shape, dtype):
        self.dtype = dtype
        host_dtype = np.uint32 if dtype == mx.uint32 else np.uint16
        if dtype not in (mx.uint32, mx.float16, mx.bfloat16):
            raise Declined("Deferred PLE unsupported packed leaf dtype")
        self.host = np.empty(shape, dtype=host_dtype)
        try:
            raw = mx.asarray(self.host, copy=False)
        except (ValueError, RuntimeError) as exc:
            raise Declined("Deferred PLE strict buffer adoption refused") from exc
        self.value = raw if dtype == mx.uint32 else raw.view(dtype)
        self.filled = False
        self.aborted = False

    def fill(self, host):
        if self.filled or self.aborted:
            raise RuntimeError("Deferred PLE packed leaf is not fresh")
        bits = host.view(np.uint16) if host.dtype == np.float16 else host
        if bits.shape != self.host.shape or bits.dtype != self.host.dtype:
            raise RuntimeError("Deferred PLE packed raw-bit geometry mismatch")
        np.copyto(self.host, bits, casting="no")
        self.host.setflags(write=False)
        self.filled = True

    def abort(self):
        self.aborted = True


@dataclass
class PackedPlan:
    layer: object
    cache: object

    def __post_init__(self):
        self.table = getattr(self.layer.ngram_embedding, "_file_backed", None)
        t = self.table
        if t is None or not t._host_assembly or t._closed or not t.shards:
            raise Declined("Deferred PLE packed requires an open host-assembly table")
        self.shard = t.shards[0]
        signature = self.shard.layout_signature
        if any(s.layout_signature != signature for s in t.shards):
            raise Declined("Deferred PLE packed refuses heterogeneous table layouts")
        # Every possible shard must have the same contract, not just the
        # shards selected by yesterday's token. Existing reader validated it.
        if (self.shard.mode != "affine" or self.shard.weight.dtype_tag != "U32"
                or self.shard.scales.dtype_tag not in ("F16", "BF16")
                or self.shard.biases.dtype_tag != self.shard.scales.dtype_tag):
            raise Declined("Deferred PLE packed requires U32 affine / matching 16-bit metadata")
        h = self.layer.hasher
        self.heads = int(h.ngram_heads)
        self.offsets = np.asarray(h.head_offsets, dtype=np.int64)
        self.sizes = np.asarray(h.head_vocab_sizes, dtype=np.int64)
        if (self.offsets.shape != (self.heads,) or self.sizes.shape != (self.heads,)
                or self.heads <= 0 or self.heads > 128 or h.context_len <= 0
                or np.any(self.sizes <= 0)
                or self.offsets[0] < 0
                or np.any(self.offsets[1:] < self.offsets[:-1] + self.sizes[:-1])
                or self.offsets[-1] + self.sizes[-1] > t.total_rows):
            raise Declined("Deferred PLE packed requires ordered disjoint head row ranges")
        if self.cache is None:
            raise Declined("Deferred PLE packed requires native per-layer state")
        self.leaves = []
        for reader in (self.shard.weight, self.shard.scales, self.shard.biases):
            shape = [self.heads, *reader.shape[1:]]
            self.leaves.append(PackedOwnedLeaf(shape, reader.mlx_dtype))
        # EXACT existing dequant/1-bit-expansion function and exact H-row
        # kernel geometry. Original uniform cohort has sorted unique rows,
        # then an identity inverse gather; retain even that gather operation.
        values = self.shard._dequantize_mlx(
            *(leaf.value for leaf in self.leaves), profile=None
        )
        inverse = mx.array(np.arange(self.heads, dtype=np.uint32))
        values = values[inverse]
        # ShardedNGramEmbedding.__call__ owns an additional compute-dtype
        # cast AFTER reader dequantization/inverse gather. JANG metadata may
        # be BF16 while the model consumes F16; preserve this exact boundary.
        output_dtype = self.layer.ngram_embedding.output_dtype
        if output_dtype is not None and values.dtype != output_dtype:
            values = values.astype(output_dtype)
        self.embedding = values.reshape(1, 1, self.heads * t.head_dim)
        self.inputs = None
        self.previous = None

    def prepare(self, ids):
        if self.inputs is None:
            raise RuntimeError("Deferred PLE packed layer not visited")
        h = self.layer.hasher
        previous = (np.asarray(self.previous, dtype=np.int64)
                    if self.previous is not None else None)
        rows = h.hash_tokens(ids, previous).reshape(-1)
        # NGramHasher hashes each head within its own positive modulo range.
        # B1/S1 therefore has H distinct ascending rows: dedup never shrinks H.
        if (rows.shape != (self.heads,) or np.any(rows < self.offsets)
                or np.any(rows >= self.offsets + self.sizes)
                or np.any(rows[1:] <= rows[:-1])):
            raise RuntimeError("Deferred PLE packed hash violated admitted unique head ranges")
        if self.table._closed:
            raise RuntimeError("Deferred PLE table closed during graph build")
        inverse, unique_count, shard_count, groups = self.table._read_host_assembled(rows)
        identity = np.arange(self.heads)
        if (unique_count != self.heads or len(groups) != 1
                or not np.array_equal(inverse, identity)
                or groups[0][0].layout_signature != self.shard.layout_signature
                or not np.array_equal(groups[0][1], identity)):
            raise RuntimeError("Deferred PLE packed cohort/dedup changed after admission")
        hosts = groups[0][2]
        payloads = []
        for host, leaf, reader in zip(hosts, self.leaves,
                                     (self.shard.weight, self.shard.scales, self.shard.biases)):
            expected_host = {"U32": np.dtype("uint32"), "F16": np.dtype("float16"),
                             "BF16": np.dtype("uint16")}[reader.dtype_tag]
            if tuple(host.shape) != tuple(leaf.value.shape) or host.dtype != expected_host:
                raise RuntimeError("Deferred PLE packed reader changed exact payload format")
            payloads.append(host)
        context = previous if previous is not None else np.full(
            (1, h.context_len), h.eos_token_id, dtype=np.int64
        )
        history = mx.array(np.concatenate([context, ids], axis=1)
                           [:, -h.context_len:].astype(np.int32))
        return payloads, history, shard_count


class PackedDeferredPLE:
    def __init__(self, layer_caches):
        self.plans = {}
        for layer, cache in layer_caches:
            if id(layer) in self.plans:
                raise Declined("Deferred PLE packed duplicate PLE layer")
            self.plans[id(layer)] = PackedPlan(layer, cache)
        if not self.plans:
            raise Declined("Deferred PLE packed no PLE layers")
        self.flushed = False

    def __enter__(self):
        if _active.get() is not None:
            raise RuntimeError("Deferred PLE scopes cannot nest")
        self.token = _active.set(self)
        return self

    def __exit__(self, kind, value, trace):
        _active.reset(self.token)
        if kind is not None or not self.flushed:
            for plan in self.plans.values():
                for leaf in plan.leaves:
                    leaf.abort()
        if kind is None and not self.flushed:
            raise RuntimeError("Deferred PLE packed graph escaped without bind")

    def bind_packed(self, layer, inputs, cache):
        p = self.plans.get(id(layer))
        if p is None or p.cache is not cache or p.inputs is not None:
            raise RuntimeError("Deferred PLE packed layer/cache ownership mismatch")
        if tuple(inputs.shape) != (1, 1):
            raise RuntimeError("Deferred PLE packed is AR B1/S1 only")
        p.inputs, p.previous = inputs, cache[2]
        return p.embedding

    def flush(self):
        if self.flushed or any(p.inputs is None for p in self.plans.values()):
            raise RuntimeError("Deferred PLE packed incomplete/repeated flush")
        # Product passes the same lazy token object to every PLE layer. Reject
        # any different input rather than implicitly synchronizing two steps.
        plans = list(self.plans.values())
        ids_array = plans[0].inputs
        if any(p.inputs is not ids_array for p in plans):
            raise RuntimeError("Deferred PLE packed PLE inputs do not share one AR token")
        ids = np.asarray(ids_array, dtype=np.int64)  # ONE prior-token wait.
        prepared = [p.prepare(ids) for p in plans]   # CPU SSD rows only.
        # No mx.eval and no float materialization: dequantization stays in
        # the original GPU graph and starts after all packed leaves are bound.
        for plan, (payloads, _, _) in zip(plans, prepared):
            for leaf, host in zip(plan.leaves, payloads):
                leaf.fill(host)
        for plan, (_, history, shards) in zip(plans, prepared):
            plan.cache[2] = history
            stats = plan.table.host_gather_stats
            stats["calls"] += 1
            stats["rows"] += plan.heads
            stats["unique_rows"] += plan.heads
            stats["shards"] += shards
            stats["layout_groups"] += 1
        self.flushed = True
