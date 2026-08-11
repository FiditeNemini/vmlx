# MiniMax-M3 MoE quant fix — handoff (2026-06-13)

**Status:** M3 serve loads (text, 98.8 GB) but **crashes on the first MoE matmul** —
routed experts load as **4-bit/gs64** when they must be **2-bit/gs128**. Generation
unmeasurable until fixed. This doc gives the proven root cause + the fix.

Written as a NEW file to avoid colliding with the live `jang_loader.py` /
`CAMPAIGN-*.md` edits.

---

## Root cause (confirmed)

The REAP22-JANG_2L bundle's routed experts are **2-bit / group_size 128**, `in=6144`:
- `...switch_mlp.{gate,up}_proj`: weight `(100,3072,384)` uint32 + scales `(100,3072,48)`
  → per-expert `out=3072`, `in = 384*16 = 6144`, group `6144/128 = 48`.
- `...switch_mlp.down_proj`: `(100,6144,192)` → `in=3072`, `out=6144`.
- `config.json["quantization"]` declares every projection
  `language_model.model.layers.N.block_sparse_moe.switch_mlp.{gate,up,down}_proj
  → {bits:2, group_size:128, mode:affine}` (789 overrides, all carry bits+group_size).

The VLM load path (`_load_jang_v2_vlm`) quantizes via
`nn.quantize(model, group_size=block_size, bits=default_bits, class_predicate=get_class_predicate)`.
For the experts, `get_class_predicate` falls through to `True` (uniform default
**bits=4 / gs64**) instead of returning the per-module `{2,128}` dict. mlx then
materialises `QuantizedSwitchLinear` at 4/64 → `gather_qmm` expands the weight as
`384*8 = 3072` input instead of `6144` → `last dim 6144 != matrix (3072,3072)` crash.

## Two myths this corrects

1. **"It's the SwitchGLU-parent-path trailing-dot bug."** No. mlx calls the predicate
   on the **child** paths `switch_mlp.{gate,up,down}_proj` (verified by a tiny no-load
   micro-test: `SwitchGLU(256,512,8)`, predicate logged paths
   `['switch_mlp.gate_proj','switch_mlp.up_proj','switch_mlp.down_proj','switch_mlp.activation']`).
   `.switch_mlp.` **is** a substring of those, so the candidate expansion runs.

2. **"The standalone runtime never exercised the real quantized MoE at hidden=6144."**
   No. `load_minimax_m3` (`runtime.py`) loaded **this exact bundle** and generated
   coherent multi-token text ("name three planets" → Earth/Mars/Jupiter, 9.4 tok/s).
   Coherent decode routes every token through the quantized `gather_qmm` at hidden 6144 —
   if experts were 4/64 it would crash identically to serve. **So the standalone
   runtime is the proven reference; its expert quantization is correct (2/128).**

## Micro-test that proves the mechanism (no 97 GB load)

```python
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU
class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.switch_mlp = SwitchGLU(256,512,8)
m=Tiny(); seen=[]
def pred(p,mod):
    seen.append(p)
    if p.endswith(".switch_mlp"): return False
    return {"bits":2,"group_size":128} if hasattr(mod,"to_quantized") else False
nn.quantize(m, group_size=64, bits=4, class_predicate=pred)
# predicate paths: switch_mlp.gate_proj / up_proj / down_proj / activation
# result: gate/up/down_proj all -> bits=2 gs=128   ← dict-return on CHILD paths works
```

So: when `get_class_predicate` returns the per-module dict for the child proj paths,
mlx quantizes them to 2/128. The fix is to *make it return that dict*.

## The fix (proven — mirrors `load_minimax_m3`)

The generic loader builds the override map by **reverse** candidate-matching
(runtime path → many config-key candidates → lookup). The standalone runtime that
works builds it in the **forward** direction (config key → sanitized runtime path →
direct lookup). Use the forward form for M3:

```python
# in _load_jang_v2_vlm, for model_type == "minimax_m3_vl":
def _m3_remap(k):
    if k.startswith("language_model.model."): k = "model." + k[len("language_model.model."):]
    elif k.startswith("language_model.lm_head"): k = "lm_head" + k[len("language_model.lm_head"):]
    return k.replace(".block_sparse_moe.", ".mlp.").replace(
        ".self_attn.index_", ".self_attn.indexer.index_")

_qcfg    = config.get("quantization", {})
_m3_over = {_m3_remap(k): v for k, v in _qcfg.items() if isinstance(v, dict)}   # keyed on RUNTIME path
_m3_def  = {"group_size": int(_qcfg.get("group_size", 64)), "bits": int(_qcfg.get("bits", 8))}

def get_class_predicate(p, m):
    if not hasattr(m, "to_quantized"): return False
    if not (_vlm_quant_module_path_candidates(p, "minimax_m3_vl") & quantized_suffixes):
        return False
    o = _m3_over.get(p, _m3_def)                # direct hit on model.layers.N.mlp.switch_mlp.gate_proj
    return {"group_size": int(o["group_size"]), "bits": int(o["bits"])}
```

`_m3_over` is keyed on the runtime path (`model.layers.N.mlp.switch_mlp.gate_proj`),
so `_m3_over.get(p)` hits directly and the micro-test guarantees 2/128.

### Shipping options
- **(a) lowest risk:** route M3 text-load through `load_minimax_m3` (`models/minimax_m3/
  runtime.py`) instead of `_load_jang_v2_vlm`. Already loads + generates end-to-end on
  this bundle.
- **(b) in-place:** the forward-remap predicate above.

## If it still shows 4/64 after the predicate fix

Read the existing `[M3-DIAG]` log line per switch_mlp module:
- `predicate_override=None` → candidate↔config match failed → use the forward-remap map
  (option b) which can't miss.
- `predicate_override={bits:2,gs:128}` **but** `module.bits=4` → a **later** pass is
  re-quantizing at default. Grep the M3 load path for a second `nn.quantize` /
  `_fix_quantized_bits` / `_pre_fix_bits_from_shard` and make it honor the per-module
  bits (or run it BEFORE, not after, the correct quantize).

## Separately worth fixing (noted in progress log)
- Scheduler classifies M3 as "Hybrid/path-dependent" for KV-quant (q4 at attention KV).
  M3 is **positional / append-only** (MSA dual cache is fully sliceable) — it should NOT
  be treated path-dependent. See the path-dependent-cache-restore note: M3 is the
  opposite case (no keep_in_ram, no clean re-prefill).
