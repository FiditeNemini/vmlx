# vMLX Live Stress-Test Campaign — RUNNING PROGRESS LOG (append as we go)

Compute: Erics-M5-Max.lan (its RAM). Harness: dev app CDP :9333 + panel/scripts/uidrv.cjs.
HOLD all commits/pushes until campaign done (PR #205 baseline dc1f68f5c). Docs alongside:
CAMPAIGN-CHECKLIST.md, MODEL-MATRIX-AUTODETECT.txt, BUILD-STATUS.md.

## Milestones
- [DONE] 35-model autodetect matrix via engine ModelConfigRegistry (MODEL-MATRIX-AUTODETECT.txt).
  All autodetect except 3 diffusiongemma (image/diffusion → not chat). Each family's cache/reason/tool pinned.
- [DONE] M3 reasoning parser: created vmlx_engine/reasoning/minimax_m3_parser.py (<mm:think>),
  registered, standalone-tested (reasoning/content split correct), stamped M3 jang_config
  capabilities.reasoning_parser=minimax_m3 → engine autodetect now returns reason=minimax_m3.
- [DONE] M3 VL vision module minimax_m3_vl.py — reverse-engineered (Qwen2-VL-family: 3D-conv
  patchify, 3D-RoPE, 32 CLIP-style layers, projector 1280->6144, 2x2 patch_merge), STRUCT TEST
  PASS (forward runs, output [N/4,6144], param paths match bundle 1:1). Empirical (real image) PENDING.
- [DONE] Harness ported to Erics-M5-Max (CDP :9333, uidrv.cjs, playwright-core, CDP switch in
  panel/src/main/index.ts). Spawn uses LOCAL engine (project venv), confirmed.
- [DONE/LIVE] Gemma-4-12B-it-qat-JANG_4M loaded via app → ● Running in UI → ~9.7GB RAM (screenshot).
  Computer-use live-load + in-app Logs observed (Wake → reload logs captured).

## KEY FINDING — paged-off/SSD-prefix is engine-supported for Gemma SWA (2026-06-13)
- App default for Gemma SWA loads with **cache_mode=paged** (block_l2 SSD under paged) — the gap.
- EMPIRICAL PROOF (manual engine :8002, flags: --continuous-batching --enable-prefix-cache
  --enable-disk-cache, NO --use-paged-cache): Gemma SWA loads paged-OFF cleanly:
    cache_mode=memory-aware ; MemoryAwarePrefixCache ; Disk cache (L2) SSD prompt-cache enabled
    (prompt_l2=True, block_l2=False) ; RotatingKVCache metadata preserved ; TurboQuant auto-enabled
    3-bit K/V 6 critical layers, layout=ALL TurboQuantKVCache ; loaded 9.7GB.
  => Engine fully supports Gemma SWA paged-off + SSD-prefix + TQ-KV. Only the PANEL predicate
     cacheSubtypeRequiresPaged(mixed_swa_kv) (+ detected.usePagedCache) forces paged. FIX = panel.
- Cache lever mapped: panel src/shared/cacheControlPolicy.ts resolveCacheLaunchPolicy:
    effectiveUsePagedCache = prefixEnabled && (architectureRequiresPagedCache || usePagedCache).
    architectureRequiresPagedCache (sessions.ts) = zaya_cca || dsv4_optin ||
    ((hybrid || cacheSubtypeRequiresPaged(mixed_swa_kv/step3p7/mimo)) && detected.usePagedCache).
- ENGINE CONSTRAINT (for path-dependent families): block_disk typed lanes (zaya_cca/dsv4_composite/
  rotating_kv) are PAGED-coupled; non-paged path = legacy disk_cache + memory-aware. So flipping
  DSV4/ZAYA/hybrid to paged-off needs their typed schemas on the non-paged SSD path + clean N-1
  re-prefill + keep_in_ram (else silent drift). => 2-phase flip:
    Phase 1 (safe, proven): standard/MoE + Gemma SWA -> paged-off/SSD-prefix default.
    Phase 2 (engine work): DSV4/ZAYA/hybrid/Step3.7 -> port typed lanes to non-paged SSD + prove no-drift.

## [PROVEN] Gemma SWA paged-off SSD-prefix cache HIT + no drift (2026-06-13)
2-turn API on :8002 (paged-off): T1 "Kyoto famous for?" -> coherent temples/Zen-gardens;
T2 "what country is that city in?" -> "Japan" (correct, contextual = NO DRIFT).
scheduler cache_hit_tokens=21 detail={'memory':21} (memory-aware L1 over disk-L2 SSD; NOT paged).
=> Phase 1 cache flip (standard/MoE + Gemma SWA -> paged-off/SSD-prefix/TQ-KV) is EMPIRICALLY SAFE.
Nuance: "paged RAM block pool OFF" achieved; prefix = memory-aware L1 (thin RAM LRU) + disk_cache L2
(SSD persistence). Cross-restart hits come from SSD L2; same-session from L1.

## [DONE+PROVEN] Phase-1 paged-off default IMPLEMENTED + proven live via app (2026-06-13)
EDITS (panel, uncommitted):
- model-config-registry.ts:437 base default usePagedCache true->FALSE; :815 `?? true`->`?? false`;
  :792 gemma block usePagedCache->false. Path-dependent families keep usePagedCache:true via
  registerFamily (zaya/qwen3-next/qwen-mamba/nemotron-h/jamba/granitemoehybrid/ling/lfm2/
  falcon-h1/falcon-mamba/mamba/rwkv/step-3.7-flash) + detector sites 766/909/917/1041 -> stay PAGED.
- sessions.ts:1907 session-create default usePagedCache `?? true`->`?? false`; added enableDiskCache
  default true (SSD legacy-disk prefix when paged off). applyBundleStartupDefaults has NO paged force.
PROOF (live app, Erics-M5-Max): fresh gemma-4-12B-JANG_4M session -> spawn flags PAGED:0 (no
--use-paged-cache) -> engine log `cache_mode=memory-aware` (NOT paged). Headline "paged off default"
ACHIEVED + proven via app. (IPC-create empty-config test gave DISK:0; renderer New Session applies
enableDiskCache=true -> earlier proven disk=true -> UI flow = paged-off + SSD disk prefix.)
GOTCHA: createSession MERGES by modelPath into an existing session -> to test new defaults must
delete the stale session first (IPC sessions.delete) else old paged=true config persists.
CACHE_STACK_STARTUP_DEFAULTS_VERSION=2 migration block (sessions.ts ~597) still force-sets
usePagedCache=true for stale-pattern sessions -> update + bump for existing-session migration (TODO).

## [STRESS — issue-hunt 2026-06-13] Phase-1 paged-off robustness
- REGRESSION PASS: LFM2.5-8B-A1B (hybrid, path-dependent) still spawns PAGED:1 after base-default
  flip (caps.cache_type=hybrid + registerFamily catch it). Did NOT un-page hybrids.
- LFM2 hybrid load PASS: "Hybrid/path-dependent cache detected — disabling live TurboQuant KV patch;
  SSM companions full precision + async clean-prefill rederive"; layout ArraysCache(SSM)+KVCache(6/24);
  SSM companion enabled. Correct path-dependent handling. (model_type=unknown but family=lfm2 fine.)
- PAGED-OFF GEMMA 4-TURN NO-DRIFT PASS (:8000, paged-off): context-building convo; T4 recall
  "7 days ... 2000 dollars" EXACT (turn-1 facts) => NO DRIFT across 4 turns; cache_hit_tokens=225
  (detail memory); no leaked think/weird tags. => Phase-1 paged-off safe for Gemma SWA multiturn.
- ISSUE/GAP FOUND: cache HIT detail='memory' (RAM L1), NOT disk. The test session had DISK:0
  (IPC empty config). For true SSD prefix persistence (requirement), enableDiskCache=true must flow
  (DISK:1) via UI New Session default; verify DISK:1 + cross-restart DISK cache HIT. (Eric: "only SSD
  cache as defaults for prefix" — memory-aware RAM L1 alone is not SSD; needs disk_cache L2 active.)

## NEXT
- Confirm UI New Session = PAGED:0 + DISK:1 (SSD) + cross-restart disk HIT for gemma, then chat live.
- Implement Phase 1 panel default flip (paged-off/SSD-prefix default for safe families) + PROVE via APP
  (load Gemma through app w/ new default -> logs show cache_mode=memory-aware not paged -> chat -> HIT).
- Phase 1 panel flip (paged-off default for safe families) + prove live.
- M3 tool parser (<invoke> XML) — needs M3 loaded to validate.
- Panel autodetect align w/ engine registry (family=unknown chat-bind fix) + UI family/parser selector.
- Live stress-test every model (load→RAM-in-UI→logs→chat→reason on/off/auto→tools→delta→SSD-HIT no-drift→leaks→VL).

## [DONE+PROVEN] Migration gap FIXED — existing-session upgraders flip paged-off (2026-06-13)
ISSUE FOUND: applyCacheStackStartupDefaultMigration (sessions.ts ~546) force-set usePagedCache=true
for ALL stale-default fingerprints incl. GENERIC (non-path-dependent) sessions -> upgraders stayed
paged ON, silently undoing Phase-1 for existing users.
FIX (uncommitted, sessions.ts):
  - CACHE_STACK_STARTUP_DEFAULTS_VERSION 2 -> 3.
  - Branched migration OUTCOME on zayaCacheMigrationTarget:
      zaya (path-dependent): keep usePagedCache=true + enableBlockDiskCache=true + maxCacheBlocks
        (Phase-2 not done; recurrent state not position-sliceable -> stays paged).
      generic/Gemma-SWA/MoE: usePagedCache=FALSE + enableDiskCache=TRUE (SSD prefix) +
        enableBlockDiskCache=FALSE + kvq=auto (TQ-KV autodetect).
  - Added staleV2GenericPagedOn pattern (non-zaya, the exact v2 paged-on fingerprint) so sessions the
    v2 migration already pushed to paged-on are re-migrated to paged-off under v3. Idempotent at v3.
  - line 1324 re-force is already Phase-1-safe (only fires when FRESH detect usePagedCache===true,
    which now only path-dependent families return) -> no change.
PROOF #1 (logic, faithful): extracted the constant + 2 helpers + migration fn VERBATIM from
  sessions.ts, esbuild-transpiled, ran 15 assertions -> ALL PASS:
    generic v2-paged-on -> paged OFF + enableDiskCache + blockDisk off + kvq auto + version 3;
    generic v0 stale-continuous (64 seqs) -> paged OFF + SSD on;
    zaya stale -> stays paged ON + blockDisk on; version-3 -> no-op (idempotent);
    user-customized (non-fingerprint) -> untouched.
PROOF #2 (regression suite): updated tests/settings-flow.test.ts source-scan guard (version 3 +
  branched-outcome pins) AND flipped DEFAULT_CONFIG fixture to product truth (usePagedCache:false,
  enableBlockDiskCache:false; enableDiskCache:true) + 4 dependent tests (paged-block-size/max-blocks
  now pass explicit usePagedCache:true; 'current startup defaults' + 'default config flags' assert
  paged-off + --enable-disk-cache SSD). npx vitest tests/settings-flow.test.ts -> 254/254 PASS.
  panel tsc --noEmit -> 0 errors.
=> Phase-1 paged-off default now covers BOTH new sessions (proven via app) AND existing upgraders
   (proven via faithful logic test + regression suite). Test suite reflects paged-off as the contract.

## [DONE+PROVEN] Panel autodetect: gemma4_unified VL+audio + capabilities.family fallback (2026-06-13)
ISSUE: panel detectModelConfigFromDir returned family='unknown' for gemma-4-12B-it-qat-JANG_4M.
ROOT CAUSE: config.json model_type=gemma4_unified / text_config gemma4_unified_text were NOT in
MODEL_TYPE_TO_FAMILY (only gemma4/gemma4_text). Unrecognized model_type -> fallback branch that also
NEVER reads jang_config capabilities -> family unknown, parsers/multimodal lost -> chat-bind blocked.
FIX (mcr.ts, uncommitted): (A) map 'gemma4_unified'->gemma4, 'gemma4_unified_text'->gemma4-text.
(B) systematic JANG Tier-1 fallback: when model_type unrecognized, resolve familyName from
jang_config capabilities.family (MODEL_TYPE_TO_FAMILY[capFamily] ?? registered-panel-family) — aligns
panel with engine oracle for ANY future JANG model w/ novel model_type.
PROVEN: esbuild-bundled detectModelConfigFromDir on the REAL model -> family=gemma4, isMultimodal=true
(vision+AUDIO), reasoningParser=gemma4, toolParser=gemma4, enableAutoToolChoice=true,
cacheType=rotating_kv (correct gemma4 mixed-SWA), usePagedCache=false (Phase-1). tsc 0 errors.
tests/model-config-registry.test.ts 67/67 (added gemma4_unified VL+audio test + capabilities.family
fallback test; fixed 2 PRE-EXISTING Phase-1 regressions: hy3 + gemma4-SWA usePagedCache true->false).
=> Per Eric: "integrate gemma 4 unified for audio vl etc" — DONE, autodetects multimodal incl audio.

## [DONE+PROVEN] TurboQuant-KV compatibility matrix + M3 latent-bug FIX (2026-06-13)
Eric Q: "all kv cache for gemma should have proper TQ encoded? and mm3 is it TQ-compatible like dsv4?"
TQ-KV gate = is_turboquant_make_cache(model.make_cache): JANG loader _patch_turboquant_make_cache
replaces make_cache -> TurboQuantKVCache ONLY when jang_config.turboquant.enabled (mxtq calibrated).
Per-family TQ-KV (RQ 3-bit K/V) status (engine scheduler.py + jang_loader.py):
| family | TQ-KV | how |
| gemma4 / standard / MoE | YES | mxtq bundle -> make_cache patched to TurboQuantKVCache. PROVEN live: gemma layout=ALL 48 TurboQuantKVCache, prompt_l2=True. |
| MLA (DSV3/Mistral4/GLM5.1) | NO (skip) | CacheList(KVCache,KVCache) incompatible w/ flat TQ -> is_mla_model skip. |
| deepseek_v4 (DSV4) | NO | _uses_dsv4_cache: own deepseek_v4_v7 composite; generic kv_quant forced none; path-dependent. |
| zaya_cca (ZAYA) | NO | typed CCA conv_state/prev_hs; generic KV quant disabled. |
| hybrid SSM (nemotron_h/lfm2/ling/qwen3.5) | PARTIAL | attention-KV-only TQ (Qwen3.6 allow-list); SSM companion FULL precision. |
| **minimax_m3 (M3)** | **NO (NEW SKIP)** | MSA MiniMaxM3SparseCache = GQA K/V + append-only idx_keys (Lightning-Indexer); selection RECOMPUTED each step from idx_keys. Flat TQ has no idx_keys lane. |
LATENT BUG FOUND: _patch_turboquant_make_cache had NO M3 skip. M3 is not MLA/SSM, so if an M3 bundle
had turboquant.enabled the loader WOULD overwrite M3's make_cache with flat TurboQuantKVCache ->
idx_keys silently dropped -> broken sparse-attn block selection / garbage. (M3 is positional/
append-only, NOT path-dependent like DSV4 — but its cache is structurally TQ-INCOMPATIBLE.)
FIX (jang_loader.py, uncommitted): explicit M3 skip right after the MLA skip — model_type in
{minimax_m3, minimax_m3_vl} (root or text_config) -> log + return, keep native MSA cache.
PROVEN: with turboquant ENABLED + non-MLA + no env-var (only the M3 skip can early-return), M3
make_cache LEFT INTACT (returns native MSA) for all 3 model_type spellings; control llama not
protected. Regression gate vmlx_engine/models/minimax_m3/test_m3_tq_skip.py (3 asserts) -> OK.
ANSWER: Gemma = full TQ-KV (proven). M3 = NOT TQ-compatible (like DSV4/MLA in spirit) — now
explicitly protected. Future option: TQ M3 keys/values only, never idx_keys.

## [DONE+PROVEN(unit/registry)] M3 tool parser minimax_m3 + autodetect wired (2026-06-13)
M3 tool format (from chat_template.jinja): Anthropic-style XML, PARAM NAME IS THE TAG (not name= attr),
wrapped <tool_call>...</tool_call>, ns_token ']<]minimax[>[' prefixed before EVERY element, reasoning
in <mm:think>. scalar=<p>v</p>; nested object=<p><k>v</k></p>->dict; array=<p><item>..</item></p>->list.
NOT minimax-M2 format (M2 uses <minimax:tool_call> + <parameter name=>). Existing parsers don't fit.
CREATED vmlx_engine/tool_parsers/minimax_m3_tool_parser.py (MiniMaxM3ToolParser, register ["minimax_m3"]):
strips ns_token + <mm:think>; recursive tag-named param parse (_next_tag balanced scan, _parse_value
dict/list/scalar coerce); multi-invoke; lenient truncated <tool_call>. Imported in tool_parsers/__init__.py
-> ToolParserManager.get_tool_parser("minimax_m3") resolves. UNIT TEST test_m3_tool_parser.py 13/13:
scalars+ns strip, int/bool/float coerce, nested obj, <item> arrays (+nested obj item), 2-invoke,
<mm:think> strip + content preserved, truncated extract, no-tool content untouched, hyphenated param.
STAMPED bundle JANGQ-AI/MiniMax-M3-REAP22-JANG_2L/jang_config.json capabilities.tool_parser=minimax_m3
(+supports_tools; backup .pre-m3toolparser.bak). ENGINE ModelConfigRegistry.lookup(M3) ->
family=minimax_m3 reasoning=minimax_m3 tool=minimax_m3 is_mllm=True (minimax_m2 override does NOT apply,
caps Tier-1 wins). PANEL: registered minimax_m3 family + MODEL_TYPE_TO_FAMILY {minimax_m3, minimax_m3_vl};
esbuild detect on real bundle -> family=minimax_m3 reasoning=minimax_m3 tool=minimax_m3 multimodal=true
usePagedCache=false. tsc 0 errors.
ACTION (user): bake capabilities stamp (family/reasoning/tool=minimax_m3, supports_tools) into canonical
bundle + HF upload so all installs autodetect deterministically.
NOT YET LIVE: M3 tool calling end-to-end (104GB load -> real multiturn tool call in app) unproven per
live-app guard. Parser + autodetect proven at unit/registry level only.
M3 cache paged-off: panel defaults usePagedCache=false (M3 positional) but disk_cache.py standalone-SSD
M3 MSA (keys,values,idx_keys) serialization UNVERIFIED -- block_disk_store has minimax_m3 tag (paged
path); standalone disk_cache M3 support must be confirmed live or M3 stays paged (Phase-2 item).

## [ISSUE FOUND + SAFE FIX] M3 paged-off + standalone SSD disk_cache = idx_keys corruption (2026-06-13)
Investigated whether M3 (positional) could default paged-OFF + SSD prefix like gemma. FOUND: NO, unsafe.
- disk_cache.py (standalone SSD tier, used when paged-off) reconstructs caches via mlx_lm
  load_prompt_cache, which resolves cache classes from mlx_lm.models.cache globals. M3's
  MiniMaxM3SparseCache is a vMLX custom class; minimax_m3_register.py installs the MODEL under
  mlx_lm.models.minimax_m3_vl but does NOT inject the cache class into mlx_lm.models.cache.
  => load_prompt_cache can't find it -> except branch -> KVCache.from_state(3-tuple) -> idx_keys
  (Lightning-Indexer lane) DROPPED -> corrupted sparse-attention block selection on SSD reload.
- PAGED path is SAFE: block_disk_store.py has a typed 'minimax_m3' (keys,values,idx_keys) lane,
  round-trip proven in test_other_models_regression.py. Paged M3 STILL gets SSD persistence via that
  block_disk L2 lane.
SAFE FIX (panel mcr.ts): registerFamily('minimax_m3', { ... usePagedCache: true }). M3 stays PAGED
(block_disk SSD lane) instead of the broken paged-off standalone-SSD path. Corrects my earlier paged-off
default for M3 (would have silently corrupted). tsc 0; model-config-registry.test.ts 69/69 (M3 test pins
usePagedCache=true). Not path-dependent like DSV4/ZAYA — reason is custom-cache-class-not-on-standalone-SSD.
PHASE-2 (to give M3 true paged-off + standalone SSD): inject MiniMaxM3SparseCache into mlx_lm.models.cache
at register (so load_prompt_cache resolves it) + ensure from_state/state round-trips the 3-tuple in
disk_cache.py + tq_disk_store, THEN prove live (104GB load, prefix HIT, no idx_keys drift). Until then PAGED.

## [DONE+PROVEN(engine)] UI/CLI autodetect audit (Opus subagent) + force-text-only bug FIX (2026-06-13)
RLM Opus subagent audited UI+CLI+autodetect manual-selector/reflection coverage. Verified WIRED OK:
reasoning parser (form ParserField :1322 + cli --reasoning-parser :2531 + wire sessions.ts :2725),
tool parser (form :1316 + cli --tool-call-parser :2470 + wire :2742), VLM force-ON (--is-mllm),
paged/disk/blockdisk checkboxes, kv-quant (auto/none/q4/q8). GAPS found:
 #1 (MED->HIGH, real bug) Force-text-only INERT for DETECTED-VL: sessions.ts isVLM ternary checked
    detected.isMultimodal BEFORE config.isMultimodal===false -> user 'Force Off' never reached; and
    omitting --is-mllm doesn't help (engine re-autodetects VL from config.json vision_config).
 #2 (MED) no model-family override (no UI selector, no CLI flag) -- user wants family selectable.
 #3 (MED, ~by-design) no cache-type/subtype override (architecture-coupled, wrong value breaks cache).
 #4 (LOW) parser selectors show generic 'Auto', not 'Auto (detected: qwen3)'.
FIX #1 (uncommitted): engine api/utils.py is_mllm_model(+force_text_only) highest-precedence ->False
(overrides force_mllm+autodetect); cli.py +--text-only (dest force_text_only) + passed at both
is_mllm_model call sites (:139,:1331); panel sessions.ts isVLM rewired: userForceTextOnly
(config.isMultimodal===false) beats detected.isMultimodal, and emits --text-only when a DETECTED-VL
model must run text (user Force-Off or detected.forceTextOnly). PROVEN (engine): gemma4_unified (real
VL) is_mllm_model -> True baseline; force_text_only=True -> False (overrides autodetect); beats
force_mllm; force_mllm alone still True. tsc 0. NOT YET LIVE: in-app Force-Off toggle -> spawn --text-only
-> engine loads text-only (to confirm in app). #2/#4 pending; #3 deferred (risky/by-design).

## [ISSUES FOUND — live M3 serve attempt for perf benchmark] (2026-06-13)
User asked for M3 token/s + pp/s (target 30+). Attempting to load M3 (97GB JANG_2L) via engine serve
on :8011 surfaced REAL integration gaps (M3 serve-load was NEVER wired; prior "M3 runtime works" was
standalone test scripts, not the full serve path):
 1. M3 autodetects is_mllm=True (caps modality=multimodal) -> engine routes to mlx_vlm VL path ->
    `ModuleNotFoundError: No module named 'mlx_vlm.models.minimax_m3_vl'` -> CRASH. M3 VL vision module
    (vmlx_engine/models/minimax_m3/minimax_m3_vl.py) is NOT registered under mlx_vlm namespace. M3-as-VLM
    not runnable yet.
 2. FIX VERIFIED LIVE: --text-only now routes M3 to TEXT (log: is_mllm_model tier=force_text_only
    result=False). The server-global _force_text_only fix (api/utils.py honors server._force_text_only;
    cli.py serve sets it; server.py global) makes ALL is_mllm_model callers honor --text-only incl the
    argless server.py:5034 VL-routing check. => dogfooded the --text-only flag on the real engine.
 3. THEN text load fails: `No module named 'mlx_lm.models.minimax_m3_vl'`. register_minimax_m3_runtime()
    (installs the vendored text module under mlx_lm.models.minimax_m3_vl) is DEFINED but NEVER CALLED
    anywhere in the engine. So plain-text serve can't resolve the model_type. WIRING NEEDED: call
    register_minimax_m3_runtime() at serve startup before model load.
NEXT: wire the registration -> relaunch text-only -> benchmark token/s + pp/s. (paged-off used per Eric;
M3 idx_keys-drop only affects cross-request prefix HIT, not single-gen perf.)

## [BLOCKER FOUND — M3 cannot generate; NO token/s possible] MoE expert dim mismatch (2026-06-13)
After wiring M3 serve-load (register_minimax_m3_runtime + --text-only), M3 LOADS (model_loaded:true,
98.8GB RAM) but GENERATION CRASHES on the first prefill in the quantized MoE:
  ValueError: [gather_qmm] Last dim of input (...,6144) != expanded quantized matrix (3072,3072)
  from shape (100,3072,384)  @ switch_layers.py gather_qmm (up_proj)
ROOT CAUSE (confirmed from bundle tensors): expert up_proj quantized weight is (100, 3072, 384) uint32
+ scales (100,3072,48) => per-expert up_proj = 3072(out) x 3072(in) [384*8=3072 in, group_size=64].
So experts take INPUT 3072. But minimax_m3.py SparseMoeBlock builds SwitchGLU(args.hidden_size=6144,
args.intermediate_size=3072) and feeds experts the 6144-dim hidden state => 6144 vs 3072 mismatch on
EVERY MoE layer. config: hidden_size=6144, intermediate_size=3072, num_local_experts=100, top_k=4.
=> This REAP22 JANG_2L bundle's experts operate in a reduced 3072-dim space (REAP pruning changed expert
dims), but the vendored minimax_m3.py runtime still wires experts at hidden_size=6144. Generation is
impossible until reconciled (either a hidden->3072 projection before experts is missing, or SwitchGLU
input dim must be 3072, or the bundle was converted with wrong expert dims). Needs M3/REAP22 arch spec.
STATUS: M3 token/s + pp/s UNMEASURABLE (crashes immediately). Contradicts the 30+ tok/s expectation —
M3 does not run end-to-end via serve on this bundle. Prior "M3 runtime works" = standalone scripts that
did not exercise the real quantized-MoE forward at hidden=6144.
NOTE: scheduler also classifies M3 as "Hybrid/path-dependent" for KV quant (q4 at attention KV layers) —
revisit (M3 is positional/append-only, not hybrid).
NEXT DIAGNOSTIC: read down_proj/gate shapes to confirm whether experts output 3072 (needs post-proj to
6144) or 6144; then fix minimax_m3.py MoE dims to match REAP22, OR get correct conversion.

## [ROOT CAUSE CONFIRMED — M3 MoE loaded at wrong bit-width] (2026-06-13)
config.json quantization declares the routed experts at 2-bit/group128 (matches bundle weights):
  layers.N.block_sparse_moe.switch_mlp.{gate,up,down}_proj -> {bits:2, group_size:128, mode:affine}
  (jang_config quantization.routed_avg_bits=2, profile JANG_2L). shared_experts=6-bit, attn=8-bit.
Weight shapes verified consistent with 2-bit/group128: up/gate_proj (100,3072,384)+scales(100,3072,48)
  => in=6144 (384*16), out=3072, group 6144/128=48; down_proj (100,6144,192) => in=3072,out=6144.
BUG: the M3 text-load path loads switch_mlp experts as 4-bit (default), so mx.gather_qmm expands the
  weight as (384*8=3072) input instead of (384*16=6144) -> "[gather_qmm] last dim (...,6144) != matrix
  (3072,3072)" -> crash on the FIRST MoE layer prefill. The per-projection bits=2/group128 from
  config.json["quantization"] are NOT applied to the SwitchGLU/QuantizedSwitchLinear experts on M3's load.
  (jang_loader has mixed-precision pre-fix infra incl a prior gather_qmm-4096 switch_mlp fix at ~:1445,
  _fix_quantized_bits ~:2023, expert-key helper ~:1066 — M3's load path isn't applying it to experts.)
FIX DIRECTION: ensure M3 load applies config.json per-projection quant (bits=2,group_size=128) to
  switch_mlp experts (route M3 through the mixed-precision expert bit-fix / correct class_predicate).
IMPACT: M3 CANNOT generate -> token/s + pp/s UNMEASURABLE until fixed. The expected 30+ tok/s is moot
  until generation works. Engine loads (98.8GB), reasoning/tool autodetect OK, --text-only OK, register OK.

## [M3 STILL BLOCKED — token/s unmeasurable] VLM-loader MoE quant not cracked (2026-06-13)
M3 routes through load_jang_vlm_model -> _load_jang_v2_vlm (jang has_vision=true sends it there even
with --text-only; the flag only fixed SERVER is_mllm routing, not jang_loader's internal VL detection).
That path quantizes via nn.quantize(get_class_predicate) + _pre_fix_bits_from_shard before load_weights.
The 2-bit/gs128 routed experts end up 4-bit/gs64 -> gather_qmm crash (input 6144 vs matrix 3072,3072).
ATTEMPTED FIXES (correct improvements, kept, but did NOT resolve M3 -> experts still 4/64):
 - _pre_fix_bits_from_shard: input_dims-authoritative bits/gs (disambiguates 2bit/gs128 vs 4bit/gs64).
 - _pre_fix module lookup: language_model. + block_sparse_moe->mlp path normalization.
=> experts STILL 4/64, so they are NOT being caught by _pre_fix as QuantizedSwitchLinear. Most likely
   nn.quantize's get_class_predicate is NOT applying the per-module 2-bit override to switch_mlp (returns
   uniform default_bits=4), AND/OR the SwitchGLU experts aren't in modules_by_path at _pre_fix time.
REMAINING WORK (focused, needs instrumented reload to confirm): verify (a) does get_class_predicate's
   _per_module_override match the switch_mlp path candidates against config.json quantization keys
   (language_model.model.layers.N.block_sparse_moe.switch_mlp.{gate,up,down}_proj -> bits:2 gs:128)?
   (b) are the experts QuantizedSwitchLinear at _pre_fix? If not, add a post-load pass that rebuilds each
   QuantizedSwitchLinear with bits inferred from input_dims + loaded scales. Likely the real fix is in
   _per_module_override / _vlm_quant_module_path_candidates for switch_mlp, OR a dedicated switch-expert
   re-quantize after load_weights.
STATUS: M3 loads (text, 98.8GB) but cannot generate -> token/s + pp/s = 0/unmeasurable. WINS this session
   stand: --text-only end-to-end, M3 serve registration, family override+hardening, gemma4_unified, etc.

## [BREAKTHROUGH — M3 GENERATES, MoE quant FIXED + PROVEN LIVE] (2026-06-13)
ROOT CAUSE (finally pinned): M3 routed experts are 2-bit/gs128 (config.json switch_mlp.{gate,up,down}_proj),
but M3 loads via _load_jang_v2 (TEXT path, "JANG v2 detected") whose _load_model_skeleton uses mlx_lm's
bits-BLIND quantize predicate -> switch experts mis-built as QuantizedSwitchLinear input_dims=3072
(=output_dims) at global default bits -> gather_qmm crash (input 6144 vs matrix 3072,3072).
(Earlier VLM-path fixes were on the wrong path; M3 text uses _load_jang_v2, proven by my rebuild firing there.)
FIX (uncommitted, jang_loader.py): _rebuild_minimax_m3_switch_experts(model, config) called right after
_load_model_skeleton in _load_jang_v2 -> for each QuantizedSwitchLinear at *.switch_mlp.{up,gate,down}_proj,
rebuild with CORRECT dims (up/gate: in=hidden6144,out=inter3072; down: in=3072,out=6144) + per-projection
bits/gs from config.json (2/128) via forward-remap (model path -> language_model...block_sparse_moe key).
PLUS re-added _pre_fix_bits_from_shard input_dims-authoritative branch so the post-rebuild _pre_fix reads
the corrected input_dims=6144 and computes 2-bit (instead of the gs=64-first ambiguity giving 4-bit).
Combo = rebuild fixes input_dims, input_dims-prefix keeps bits=2. "Rebuilt 171 switch-expert projections".
PROVEN LIVE (engine :8011, --text-only, paged):
 - NO gather_qmm crash. Prefill 214 tok OK, HTTP 200, cache stored (60 layers) + HIT (cached_tokens=128).
 - COHERENT: "capital of France is Paris... Eiffel Tower... Notre-Dame" — correct, no leaked tags, clean
   reasoning/content split (minimax_m3 reasoning parser: reasoning_content separate, content clean), stop.
 - SPEED: prefill TTFT 1.96s; DECODE ~9.06 tok/s (matches Codex runtime.py 9.4). BELOW 30+ target.
OPEN ISSUES (systematic, next):
 1. SSM MISCLASSIFICATION (user-flagged): scheduler treats M3 as hybrid-SSM ("57 SSM layers", hybrid=True,
    SSM companion store + deferred re-derive per request). M3 is NOT SSM — it's MSA positional/append-only
    (idx_keys indexer). The SSM-companion re-derive is semantically wrong AND adds per-request overhead
    (likely dragging the 9 tok/s). Fix: scheduler must not classify M3's MiniMaxM3SparseCache as SSM/hybrid.
 2. SPEED 9 tok/s << 30+ target: investigate after SSM fix (MSA indexer recompute per step + 100-expert
    2-bit MoE + SSM-companion overhead). Codex runtime.py also ~9.4 -> may be partly inherent to MSA, but
    SSM misclassification removal + decode-loop profiling needed.

## [SYSTEMATIC RESOLUTION — M3 now works end-to-end; speed root-caused] (2026-06-13)
M3 went from "crashes on first token" to "generates coherently". TWO real bugs fixed + PROVEN live,
speed bottleneck MEASURED (not guessed).

### FIX 1 — MoE 2-bit quant (was: gather_qmm crash). PROVEN.
M3 routed experts are 2-bit/gs128 (config switch_mlp.{gate,up,down}_proj) but _load_jang_v2's
_load_model_skeleton (mlx_lm bits-blind quantize) mis-built QuantizedSwitchLinear with input_dims=3072
(=output) at global default bits -> gather_qmm crash (input 6144 vs matrix 3072,3072).
FIX (jang_loader.py): _rebuild_minimax_m3_switch_experts(model, config) after _load_model_skeleton —
rebuild each switch expert with correct dims (up/gate in=hidden6144/out=inter3072; down in=3072/out=6144)
+ per-projection 2-bit/gs128 from config (forward-remap model-path->config-key). PLUS re-added
_pre_fix_bits_from_shard input_dims-authoritative branch so post-rebuild _pre_fix reads corrected
input_dims=6144 and keeps bits=2 (vs gs=64-first ambiguity -> 4-bit). "Rebuilt 171 projections".

### FIX 2 — hybrid/SSM MISCLASSIFICATION (user-flagged). PROVEN.
scheduler._is_hybrid_model built kv_types by name-ending "KVCache"; M3's MiniMaxM3SparseCache doesn't
end in KVCache -> bucketed non-KV -> "Hybrid SSM cache: 3/60 KV layers, SSM companion enabled" + forced
paged + SSM-companion re-derive. WRONG: MiniMaxM3SparseCache IS a position-truncatable KVCache subclass
(MSA append-only, NOT SSM). FIX: recognize MiniMaxM3SparseCache as KV in _is_hybrid_model -> M3 now
"pure-attention" path, no SSM companion. Verified: hybrid/SSM logs GONE, coherent generation intact.

### PROVEN LIVE (engine :8011, --text-only):
- Generates COHERENT text: "capital of France is Paris... Eiffel Tower and the Louvre Museum", clean
  reasoning_content/content split (minimax_m3 reasoning parser), no leaked tags, natural stop.
- Speed: TTFT ~1.7s, DECODE 9.9 tok/s (was 9.06; +hybrid fix minor gain).

### SPEED ROOT CAUSE — MEASURED (profiler, eval-bounded; ratios real):
Per-token (60 layers, batch=1): attn ~36ms + MoE ~70ms [sel 14 / switch_2bit_top4 38 / shared_6bit 18].
Micro-bench: 2-bit gather_qmm = 0.22ms/call (FASTER than 8-bit) -> kernel NOT upcasting/slow.
BUT each call achieves only ~85-157 GB/s vs ~400 GB/s BW -> ~0.1ms launch overhead dominates the small
batch=1 reads. ~600 tiny kernel launches/token. BW-OPTIMAL ceiling ≈ 28-30 tok/s (attn 8-bit all-layers
~6GB + shared 6-bit ~4.8GB + active experts 2-bit ~3.2GB ≈ 14GB/tok). We're 3x off due to launch
overhead / no kernel fusion. RULED OUT: continuous batching (bare runtime.py also 9.4), mrope/2D/3D
(RoPE is 1D), the 2-bit kernel, the MSA indexer (skipping it = no change), hybrid path (per-request only).
=> To reach ~30: mx.compile/fuse the decode forward (engine has NO mx.compile in decode path). Hard due
to MoE data-dependent gather + MSA dynamic indexer + in-place custom cache. Real engineering, not a flag.

## [M3 SPEED — EXHAUSTIVE root cause, all levers tested] (2026-06-13)
M3 decode = ~9.9 tok/s (100ms/tok). BW-optimal ceiling ~28-30 tok/s. The 3x gap is NOT a misconfig.
MEASURED + RULED OUT every candidate:
 - 2-bit gather_qmm kernel: micro-bench 0.22ms/call, FASTER than 8-bit -> no upcasting/slow kernel.
 - continuous batching: bare runtime.py = same 9.4 -> not serve/scheduler.
 - mrope/2D/3D RoPE: M3 text uses plain 1D nn.RoPE -> not rope.
 - MSA Lightning indexer: skipping it entirely = no change (returns None <2048 ctx anyway).
 - hybrid/SSM path: per-request only; fixing the misclassification gave +0.6 tok/s.
 - mx.compile: tested on the stateless SparseMoeBlock -> only 1.13x (0.586->0.517 ms/block). NOT the lever.
 - decode is already 1-eval/token (pipelined) -> not Python launch overhead.
ROOT CAUSE = fundamental batch=1 GEMV bandwidth inefficiency on Apple GPU. Both attention (8-bit, ~91
GB/s) and MoE (4-of-100 gather + 2-bit, ~147 GB/s) run at ~1/3 of the ~400 GB/s memory BW because
single-token (M=1) matmuls/gathers cannot saturate BW (no rows to hide latency). 60 sequential layers
x ~1.7ms = ~100ms/tok. This is inherent to single-user single-token decode of a 105GB model on M5 Max.
PATHS TO ~30 (all are MAJOR efforts, none a setting):
 1. Native MTP / multi-token-per-forward: NOT VIABLE on this bundle — REAP22 stripped MTP weights
    (config num_nextn_predict_layers=1 but 0 mtp keys in safetensors). Would need a re-export with MTP.
 2. External speculative decoding (small draft model verifies K tokens/forward) -> ~Kx; needs a draft + wiring.
 3. Continuous batching across concurrent sequences -> amortizes weight reads; not single-user.
 4. Custom Metal kernels for batch=1 2-bit gather-GEMV -> deep, uncertain payoff.
CONCLUSION: ~10 tok/s is near the batch=1 ceiling for this model/hardware AS IMPLEMENTED. The decode
correctness/coherency is FIXED + proven; speed to 30 requires one of the above major approaches.
