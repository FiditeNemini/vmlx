# MiniMax-M3 vMLX — Build Status (live; work-in-tree, DO NOT commit/push until ALL done)

Working on **Erics-M5-Max** (`~/mlx/vllm-mlx`). 104.5 GB model at
`~/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP22-JANG_2L`. Branch `feat/minimax-m3-runtime`
(commit dc1f68f5c already pushed = PR #205 baseline). **Per Eric: no further commits/pushes
until every deliverable below is complete.**

## Deliverables
| # | Item | Status |
|---|------|--------|
| 1 | Audit/verify the existing M3 runtime + 4-tier cache | ✅ DONE — all 4 tests PASS on real 97GB model (logit diff 0) |
| 2 | PR so the team can help (baseline) | ✅ DONE — PR #205 (no further pushes until all done) |
| 3 | **VL wrapper** `minimax_m3_vl.py` | ⏳ IN PROGRESS |
| 4 | Live in-app verification (dev app, real model) | ⏳ TODO |
| 5 | Release integration (register/gen-config/capability) | ⏳ TODO |

## #3 VL wrapper — spec gathered
- `model_type=minimax_m3_vl`, arch `MiniMaxM3SparseForConditionalGeneration`,
  `image_token_index=200025`, `image_seq_length=576`, projector_hidden_size=6144, projector act gelu.
- **vision_config:** `clip_vision_model`, hidden 1280, 32 layers, 16 heads, intermediate 5120,
  patch 14, image_size 2016, projection_dim 6144, **position_embedding_type=rope, rope_mode=3d**,
  theta 10000, act gelu, ln_eps 1e-5. Compression: **patch_merge**, spatial_merge_size=2, temporal 2.
- **Weight layout (standard HF CLIP naming):**
  `vision_tower.vision_model.embeddings.patch_embedding`, `.pre_layrnorm`,
  `.encoder.layers.N.{self_attn.{q,k,v,out}_proj, layer_norm1/2, mlp.fc1/fc2}` (quantized: weight/scales/biases),
  then `patch_merge_mlp.linear_N`, `multi_modal_projector.linear_N`.
- **Dependency / open question:** upstream `modeling_minimax_m3_vl.py` is NOT in the bundle
  (only configuration + image/video processors). Need the vision-tower forward reference
  (3D-RoPE detail, patch_merge exact, projector) — from mlx_vlm if it has it, else upstream HF.

### Reference for the port (oracle)
- Bundle config states classes "intentionally mirror **`sglang.srt.configs.minimax_vl`**" → the
  vision FORWARD reference is **SGLang** (open source, `sgl-project/sglang`). SGLang has
  `python/sglang/srt/models/clip.py` (CLIP tower; vision attn = SGLang `VisionAttention`).
  The M3-VL composer (vision_tower → patch_merge → projector → splice → text backbone) is the
  SGLang MiniMax-VL model (config name `minimax_vl`; exact model filename TBD — not under an
  obvious `minimax*` name in srt/models, likely `minimax_vl_01.py`/recent; confirm before porting).
- NOT available: no `modeling_minimax_m3_vl.py` in the bundle; mlx_vlm has no minimax_m3;
  jang converter only passes vision tensors through (8-bit). mlx_vlm CLIP templates to mirror:
  `models/{pixtral,llava_bunny,gemma3,gemma4,hunyuan_vl}/vision.py`.

### REVERSE-ENGINEERING RESULT (2026-06-13) — M3 vision = Qwen2-VL-family
Reference is NOT published (HF repo = config+processors only; GitHub = docs only; transformers 5.7
has only minimax_m2; sglang main has no minimax VL; original source deleted). BUT reverse-engineered
definitively from the bundled image_processor + tensor shapes:
- **Image processor is byte-for-byte Qwen2-VL**: `smart_resize(factor=patch*merge=28, max_pixels=451584)`,
  `temporal_patch_size=2`, `merge_size=2`, patch flatten → `pixel_values[num_patches, 3*2*14*14=1176]`
  + `image_grid_thw=[t,h,w]`. mean/std = CLIP's.
- **patch_embedding.weight = (1280,3,2,14,14)** → 3D conv patchify (Qwen2-VL style). NO position_embedding,
  NO class_embedding, NO post_layernorm → **3D-RoPE**, no CLS. `pre_layrnorm` only.
- Encoder: 32 layers, **CLIP naming** (`self_attn.{q,k,v,out}_proj` w/ bias, `mlp.fc1(1280->5120)/fc2`,
  `layer_norm1/2`), hidden 1280, 16 heads (head_dim 80), gelu. 8-bit affine quant.
- **multi_modal_projector**: linear_1 (1280->6144) -> gelu -> linear_2 (6144->6144).
- **patch_merge_mlp**: linear_1 (24576->6144) -> gelu -> linear_2 (6144->6144); 24576 = 4*6144 = 2x2
  spatial merge AFTER projection. (Confirm projector-then-merge order empirically.)
- Splice projected+merged image embeds at `image_token_index=200025`; feed M3 text backbone.

**Port template = mlx_vlm `qwen2_5_vl/vision.py`** (3D patchify + 3D rope from grid_thw + window/full
attn + merger). Adapt: CLIP-style per-layer naming, 1280/16/5120 dims, separate projector + patch_merge.
**Validation = empirical** (real image -> coherent description), since no numerical oracle exists.

### VL build sub-steps
- [x] Identify architecture (Qwen2-VL-family) + dims + input path. DONE.
- [x] **Wrote `vmlx_engine/models/minimax_m3/minimax_m3_vl.py`** (working tree, UNCOMMITTED):
      3D-conv patch_embed, 3D-RoPE(h,w), 32 CLIP-style layers (LayerNorm/separate-qkv+bias/gelu,
      full attn per image via cu_seqlens), pre_layrnorm, projector(1280->6144->6144),
      2x2 merge + patch_merge(24576->6144->6144), + Qwen2-VL image preprocessing (smart_resize).
- [x] **STRUCTURAL TEST PASS** (test_m3_vision_struct.py, random weights, no 104GB load):
      forward runs end-to-end; output shape [N/4, 6144] correct; **all param paths match the bundle
      weight names 1:1** -> will load the real quantized weights.
- [ ] Quantized weight load: nn.quantize(vision stack, 8-bit) + sanitize Conv3d weight
      (1280,3,2,14,14)->MLX layout (transpose like qwen2_5_vl sanitize), load real weights.
- [ ] Wire splice: add inputs_embeds path to minimax_m3.py Model (currently takes token ids);
      compute text embeds, replace image_token_index=200025 positions with vision embeds, forward
      through MSA dual cache (image turn = full prefill; text continuation reuses cache).
- [ ] **EMPIRICAL VALIDATION (the only oracle)**: load real weights, feed /tmp/vl-test.png (red
      circle + 42) or a natural image -> confirm coherent description; iterate on rope/order/mask if
      garbled. Then confirm cache HIT on a follow-up text turn.
- NOTE: numerics (rope detail, projector-then-merge order, attn mask) NOT yet validated — structure
  is sound but coherence must be empirically confirmed before declaring VL "working".

## Final consolidation (AFTER all M3 done, per Eric)
Bring into ONE vMLX Python PR: M3 (text PR #205 + VL) + the paused vMLX fixes (PR #204 harness/
protocol) + the paged-cache-default-OFF / SSD-prefix / TQ / VL / reasoning / tool matrix
cross-checks (docs/LIVE-APP-TESTING-*.md). Hold all commits/pushes until then.
- [ ] `minimax_m3_vl.py`: CLIP-3D-RoPE vision tower (patch embed → pre_layrnorm → 32 encoder layers
      → feature select `full` @ layer -1).
- [ ] patch_merge (spatial_merge_size=2) + multi_modal_projector (→6144, gelu).
- [ ] Splice image embeds at `image_token_index=200025` into text embeds; feed text backbone.
- [ ] register into mlx_vlm/mlx_lm `minimax_m3_vl` namespace (mirror gemma4_unified_register).
- [ ] Validate vs reference (numeric) → then real-image decode.

## Audit evidence (already captured, #1)
tier-micro PASS · regression PASS · cache-roundtrip PASS (logit diff 0) · e2e L1+L2 PASS (maxdiff 0).
Run: `.venv/bin/python vmlx_engine/models/minimax_m3/test_*.py [<bundle>]`.
