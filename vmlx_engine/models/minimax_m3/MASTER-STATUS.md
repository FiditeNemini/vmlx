# vMLX Engine Campaign — MASTER STATUS (all work, done/not-done)
Compute: erics-m5-max.local. Branch feat/minimax-m3-runtime (PR #205 baseline dc1f68f5c). HOLD all commits.
All edits uncommitted. Detail in vmlx_engine/models/minimax_m3/CAMPAIGN-PROGRESS-LOG.md + project memory.

## A. PAGED CACHE OFF / SSD PREFIX DEFAULT
- [DONE+PROVEN] Phase-1 paged-OFF default for generic/Gemma-SWA/MoE: panel mcr.ts (base usePagedCache
  false; gemma mixed-SWA false) + sessions.ts (create default ?? false, enableDiskCache true). Live app:
  PAGED:0 DISK:1 prompt_l2=True, TurboQuantKVCache all layers, cross-restart SSD persist (entries=2).
- [DONE+PROVEN] Migration VERSION=3 branched: generic upgraders -> paged-off+SSD; zaya/path-dependent
  stay paged. 15 verbatim-fn asserts + settings-flow 254/254 + tsc 0.
- [DONE] Path-dependent families (zaya/qwen3-next/qwen-mamba/nemotron-h/jamba/ling/lfm2/falcon-h1/mamba/
  rwkv/step-3.7-flash) correctly STAY PAGED:1 (regression-checked LFM2 hybrid).
- [PHASE-2 NOT DONE] DSV4/ZAYA/hybrid/Step3.7 paged-OFF needs typed block-disk lanes ported to standalone
  SSD disk_cache + clean N-1 reprefill + no-drift proof. Currently stay paged (correct interim).
- [PHASE-2 NOT DONE] M3 paged-off: standalone disk_cache drops idx_keys (mlx_lm load_prompt_cache can't
  resolve MiniMaxM3SparseCache). M3 stays PAGED (block_disk minimax_m3 lane safe). [UPDATE: M3 now KV-
  classified, may be memory-aware-able — needs cross-restart SSD-HIT proof].

## B. SSD DISK / TQ KV CACHE
- [DONE+PROVEN] SSD disk_cache (prompt-cache L2) is the paged-off prefix tier; gemma cross-restart persist.
- [DONE] TQ-KV (TurboQuantKVCache, RQ) per-family matrix: gemma/standard/MoE=YES (proven 48-layer);
  MLA/DSV4/ZAYA=NO; hybrid=attn-KV-only; M3=NO (MSA idx_keys incompatible -> explicit TQ make_cache skip
  test_m3_tq_skip.py).
- [NOT DONE] Per-family SSD prefix HIT + good-TTFT live proof for EACH of hybrid/SSM/CCA/CSA/SWA/gemma-SWA
  (only gemma-SWA proven live; others need per-family live HIT+no-drift).

## C. HYBRID DERIVATION / SSM COMPANION
- [DONE prior] SSM companion store + async clean-prefill re-derive (gpl skip for thinking models).
- [DONE] M3 hybrid MISCLASSIFICATION fixed: _is_hybrid_model now recognizes MiniMaxM3SparseCache as KV
  (was tagged SSM "3/60 KV layers"); M3 -> pure-attention path. Coherency intact.
- [NOT DONE] CCA (zaya_cca_v1) / CSA / HCA (DSV4 deepseek_v4_v7) path-dependent: clean N-1 reprefill +
  cache_key_override + keep_in_ram + TQ-KV off -> NO-DRIFT live proof per family. NOT live-verified.

## D. AUTODETECT / FAMILY / REASONING / TOOL PARSER (UI+CLI)
- [DONE+PROVEN] 35-model autodetect matrix (engine ModelConfigRegistry).
- [DONE+PROVEN] gemma4_unified VL+audio autodetect (panel map + capabilities.family Tier-1 fallback);
  tests 69/69; engine+panel both resolve gemma4 multimodal.
- [DONE+PROVEN] M3 reasoning parser minimax_m3 (<mm:think>) + TOOL parser minimax_m3 (tag-named XML,
  13/13 unit) created+registered+bundle-stamped; engine+panel autodetect minimax_m3 family/reasoning/tool.
- [DONE+PROVEN] #1 force-text-only: --text-only flag + server-global _force_text_only (all is_mllm callers).
- [DONE+PROVEN] #2 model-family override: engine registry force-family + cli --model-family + UI selector
  + wiring + 4 tests; hardened (parser contract transfers).
- [DONE] #4 parser-reflection polish (ParserField 'Auto (detected: X)').
- [DEFERRED] #3 cache-type override (architecture-coupled, risky).
- [NOT DONE — user gate] LIVE-CONFIRM #1/#2/#4 in real app via CDP (dev-app restart to pick up sessions.ts
  main-process rebuild, then drive gemma: Force-Off->--text-only, family override, parser reflection).

## E. GEMMA 4 / QAT
- [DONE] gemma-4-12B-it-qat-JANG_4M: live-loaded, autodetect gemma4 multimodal, paged-off SSD prefix +
  TurboQuantKVCache proven, 4-turn no-drift, coherent.
- [NOT DONE] Other Gemma4 (E2B/E4B/26B-A4B/31B) x JANG_4M/MXFP4/MXFP8/QAT live matrix; audio/VL real
  image+audio (BLOCKED by P0 VL Stream(gpu,0) bug — Gemma4 media on executor thread w/o default gpu stream).

## F. MINIMAX-M3 (was completely broken -> now generates)
- [DONE+PROVEN] M3 serve loads + generates COHERENT text (capital-of-France test, clean reasoning/content).
  Fixes: register_minimax_m3_runtime wired; MoE 2-bit quant rebuild (_rebuild_minimax_m3_switch_experts +
  _pre_fix input_dims); hybrid->KV reclassification; --text-only.
- [DONE] M3 cache type/attention confirmed: MSA dual (keys,values,idx_keys), positional, TQ-KV off.
- [DONE+PROVEN] SPEED checkpoint in PR #205: affine-2 SwitchGLU Metal fast path moved model-forward
  decode from ~10.3 tok/s to ~22.3 tok/s warm; runtime.generate measured ~17.8 tok/s streamed and
  ~19.7 tok/s non-streamed. PR head cc436a54d.
- [IN PROGRESS] Native MTP = bundled EAGLE3 sidecar, not config-declared MiniMax MTP tensors. Local/HF
  bundle has eagle3_runtime.safetensors + eagle3_config.json. vMLX now detects this as native_mtp
  method=eagle3 and has draft loader + return_aux taps + MSA rollback helper; live verify/accept loop
  still NOT wired. See M3-EAGLE3-NATIVE-MTP-HANDOFF.md.
- [NOT DONE] M3 VL empirical (real image) — M3-as-VLM crashes (mlx_vlm.models.minimax_m3_vl missing).
- [NOT DONE] M3 tool-calling live multiturn (parser unit-proven, not live).

## G. OPEN P0 / BLOCKERS
- [P0] VL Stream(gpu,0): ALL VL image input fails (Gemma4 media on executor thread). Blocks VL live proof.
- [NOT DONE] Per-model live stress-test EVERY model in app w/ logs (reasoning on/off/auto, multiturn tools,
  content+delta, SSD HIT no-drift, leaks, VL image+audio).
- [PENDING] Consolidate all into ONE vMLX Python PR (HOLD until done).
