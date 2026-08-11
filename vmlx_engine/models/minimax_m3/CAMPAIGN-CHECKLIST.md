# vMLX — Extensive Live Stress-Test Campaign (ALL models) — Master Checklist

Goal: ONE consolidated vMLX Python PR (engine + UI + app + streaming/API) where EVERY model in
the folder loads with its OWN correct settings, autodetect + manual selection both work, the
correct per-family prefix-cache is enforced + reflected in the UI, paged RAM cache is OFF by
default (SSD prefix cache default), TQ-KV encode/decode for applicable models, and it's all
LIVE-tested in the real dev app (load → RAM-in-UI → chat as a real user → no leaks). Be
systematic; never assume; fix any mismatch and prove it. Hold commits until all done.

Compute: **Erics-M5-Max.lan** (use its RAM). Harness: dev app CDP :9333 + panel/scripts/uidrv.cjs.

## A. Per-family settings (engine autodetect oracle = ModelConfigRegistry) — see MODEL-MATRIX-AUTODETECT.txt
35 models classified; all autodetect except 3 diffusiongemma (image/diffusion → not chat).

## B. Per-family CACHE contract + path-dependence (CRITICAL — never assume)
| Family | Cache | Path-dependent? | Stress-test must prove |
|---|---|---|---|
| **minimax_m3** | MSA dual `(keys,values,idx_keys)` append-only, block=pos//128 | **NO** (fully positional/sliceable) | bit-identical reconstruct (✓ done); no keep_in_ram / no clean-reprefill needed |
| **deepseek_v4** (DSV4-Flash) | `DeepseekV4Cache`: SWA local + CSA + HCA compressed pools + incomplete-tail; schema `deepseek_v4_v7` | **YES** (compressor-output-dependent; can't trim by position) | prefix-HIT produces NO semantic drift on long gen (compare hit vs fresh); needs clean N−1 re-prefill (`_prefill_for_prompt_only_cache`) + `cache_key_override` (L2 keys on clean tokens) + `keep_in_ram=has_dsv4_cache_data` + generic TQ-KV FORCED OFF; no async-L2 race (paged table populated but reconstruct→None) |
| **zaya_cca** (ZAYA) | `CacheList(KVCache, ArraysCache)` w/ CCA conv_state+prev_hs; schema `zaya_cca_v1` | **YES** (recurrent state accumulates, not sliceable) | same fix pattern as DSV4 (clean re-prefill + cache_key_override + keep_in_ram=has_zaya_cca_cache_data + TQ-KV off); no drift on hit |
| **gemma4** (SWA) | mixed `RotatingKVCache`+full `KVCache`; subtype `mixed_swa_kv` | partial | preserve ALL config KV-head counts (asym full vs SWA); SSD prefix HIT + good TTFT |
| **nemotron_h**, **lfm2** | hybrid SSM + companion (`nemotron_h_ssm_attention`, `lfm2_moe_hybrid_ssm`) | YES (SSM cumulative) | KV + SSM companion both persist to SSD; hit correct |
| **step3p7** | `step3p7_full_sliding_kv` | partial | full+sliding KV preserved |
| **qwen3_5**, **ling** | hybrid | YES | companion persists |
| **minimax** (M2.7), standard/MoE | plain KV | NO | TQ-KV RQ encode/decode round-trips through SSD |

## C. Gaps to FIX (then prove live)
1. M3 reasoning parser (`<mm:think>`) — ✅ DONE (created vmlx_engine/reasoning/minimax_m3_parser.py,
   registered, stamped M3 jang_config capabilities.reasoning_parser=minimax_m3, autodetect verified).
2. M3 tool parser — ⏳ M3 format `<tool_call><invoke name="X">…args(xml)…</invoke></tool_call>`;
   match to an existing tool parser (xml_function / hermes / step3p5) or create minimax_m3; stamp it.
3. Panel autodetect aligns with engine registry — ⏳ (panel returned family=unknown for gemma JANG_4M
   while engine gets gemma4 → blocked chat-bind). Make panel read jang_config capability stamp.
4. UI manual selector for family / reasoning-parser / tool-parser — ⏳ (so autodetect AND override work).
5. Cache policy default: paged OFF / SSD prefix ON per family; TQ-KV for jangtq models
   (M2.7/Step-3.7/Nemotron/Ling have mxtq dicts); affine elsewhere; DSV4/M3 own caches; UI reflects.

## D. Per-model LIVE stress-test checklist (real dev app, as a user) — run for EVERY model
- [ ] Loads via local engine (.venv) → **visible loaded in RAM in the UI** (Server tab Running + mem)
- [ ] Autodetect correct (family/cache_type/cache_subtype/reasoning/tool) AND manual select works
- [ ] gen-config defaults from model (no fake fallbacks like DEFAULT_BOUNDED_TOP_K)
- [ ] Multiturn chat coherent; content + streaming delta clean
- [ ] Reasoning on/off/auto toggle proper; clean reasoning separation; NO leaked think tags
- [ ] Tool parser: multiturn native-tool calling without issue (enable tools in UI)
- [ ] Prefix cache = SSD (NOT paged); HIT confirmed + good TTFT; correct per-family cache type
- [ ] **Path-dependent families (DSV4/ZAYA/hybrid): prove NO semantic drift on prefix HIT (long gen)**
- [ ] TQ-KV RQ encode/decode round-trips (applicable models)
- [ ] VL models: real image AND audio (drag-drop / image-audio button) → coherent
- [ ] UTTERLY no leaks / thinking tags / weird chars / bad spacing
