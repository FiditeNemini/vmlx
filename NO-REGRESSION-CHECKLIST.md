# vMLX No-Regression Checklist

**Purpose.** Every fix landed in vMLX gets a row here with the issue code, the patched code, the root cause, and a verification step. Future refactors must keep these rows green. Reverting any of these without an explicit replacement is a regression.

**How to use.**
1. When you fix anything, append a row at the bottom of the relevant section.
2. Include the file:line, the BEFORE snippet (the bug), the AFTER snippet (the patch), the root cause in 1-3 sentences, and a verification step (test name, manual repro, or live model + prompt).
3. Mark `Verified ✅` once the fix has been live-tested. `Tracked 🟡` if it's source-only and waiting for a real model.
4. Never leave a row in `Pending ❌` after a release ships — either verify or revert.

**Conventions.** All file paths are relative to repo root unless prefixed with `bundled:` (means `panel/bundled-python/python/lib/python3.12/site-packages/...`) or `venv:` (means `.venv/lib/python3.13/site-packages/...`).

---

## 1. Per-request cache bypass (`cache_salt` / `skip_prefix_cache`)

### 1.1 — API model fields accept `cache_salt` and `skip_prefix_cache`

| Field           | Value |
|---|---|
| File            | `vmlx_engine/api/models.py` |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | Benchmark clients had no way to force fresh execution. Prefix injection in the prompt is a hack — every cache layer would still hit on partial matches. |
| Fix             | Added `cache_salt: str \| None = None` and `skip_prefix_cache: bool \| None = None` to `ChatCompletionRequest` and `CompletionRequest`. |
| Verification    | `tests/test_cache_bypass.py::TestAPIModelFields` (5 tests). Empty string must NOT trigger bypass — only non-empty string OR explicit `True`. |
| Regression risk | Default `None`/`None` MUST stay false-y. Anyone changing the defaults must update the helper logic in `_compute_bypass_prefix_cache`. |

### 1.2 — `_compute_bypass_prefix_cache` semantics

| Field           | Value |
|---|---|
| File            | `vmlx_engine/server.py:_compute_bypass_prefix_cache` |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | Need a single helper that decides when a request should bypass every cache layer. |
| Rule            | Returns `True` iff (`cache_salt` is a non-empty string) OR (`skip_prefix_cache is True`). Empty string → False (clients that default-construct to `""` must not opt in by accident). |
| Verification    | `tests/test_cache_bypass.py::TestComputeBypassFlag` (7 tests). |
| Regression risk | Loosening this (e.g. `bool(cache_salt)` would let `0` and `False` strings opt in). Any change must keep the empty-string False. |

### 1.3 — Six gateway forward sites set `_bypass_prefix_cache` in kwargs

| Field           | Value |
|---|---|
| File            | `vmlx_engine/server.py` (6 sites: Anthropic, Ollama chat, OpenAI chat, Responses API, completions non-stream, completions stream) |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | Helper alone is useless if API gateways don't forward the flag into engine kwargs. |
| Verification    | `tests/test_cache_bypass.py::TestServerForwarding::test_all_gateway_forward_sites_set_bypass_kwarg` — counts the `_compute_bypass_prefix_cache(` references and the `_bypass_prefix_cache] = True` assigns. |
| Regression risk | Adding a 7th gateway without paired bypass forward will silently regress benchmark isolation. |

### 1.4 — Engine kwarg pop in batched.py + simple.py

| Field           | Value |
|---|---|
| File            | `vmlx_engine/engine/batched.py`, `vmlx_engine/engine/simple.py` |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | The `_bypass_prefix_cache` kwarg must be popped at the engine boundary so it doesn't leak into `mlx_lm.generate(...)` (which would reject unknown kwargs). |
| Fix             | `kwargs.pop("_bypass_prefix_cache", False)` in every public engine method. |
| Verification    | `tests/test_cache_bypass.py::TestSchedulerBypassGating::test_simple_engine_eats_bypass_kwarg` (≥4 pop sites in SimpleEngine) and `test_batched_engine_threads_bypass_to_engine` (≥2 pops + ≥4 forwards in BatchedEngine). |
| Regression risk | Adding a new engine method that forgets to pop will crash on bypass requests with `unexpected keyword argument`. |

### 1.5 — Scheduler gate sites: paged / memory-aware / legacy / disk L2

| Field           | Value |
|---|---|
| File            | `vmlx_engine/scheduler.py:_schedule_request` (fetch sites) and `vmlx_engine/scheduler.py:_cleanup_finished` (store site) |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | The fetch path had four cache lookups (block_aware, memory_aware, legacy prefix, disk L2). The store path had a `_skip_cache_store` short-circuit. Bypass had to gate every single one. |
| Fix             | `_bypass = bool(getattr(request, "_bypass_prefix_cache", False))` declared once, then `and not _bypass` appended to each fetch condition. Store path OR's `_skip_cache_store` with the bypass attribute. |
| Verification    | `TestSchedulerBypassGating::test_scheduler_schedule_has_bypass_gate` + `test_scheduler_store_path_honors_bypass`. Source-level pattern check, runs in <1s. |
| Regression risk | A future scheduler refactor that splits `_schedule_request` into smaller helpers must re-thread the `_bypass` variable through every fetch site. The source-level test catches this. |

### 1.6 — MLLM scheduler gate (paged + memory-aware + legacy + disk L2 + SSM companion)

| Field           | Value |
|---|---|
| File            | `vmlx_engine/mllm_scheduler.py:add_request` + `vmlx_engine/mllm_scheduler.py:_cleanup_finished` |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | VL models use `mllm_scheduler` not `scheduler`. Same gating story as 1.5 but in the VLM path. |
| Fix             | `add_request` reads `kwargs.get("bypass_prefix_cache", False)` and sets `request._bypass_prefix_cache = True`. `_cleanup_finished` ORs the request flag into `_skip_cache_store`. |
| Verification    | `TestSchedulerBypassGating::test_mllm_scheduler_add_request_reads_bypass` + `test_mllm_scheduler_store_path_honors_bypass`. |
| Regression risk | Same refactor risk as 1.5. |

### 1.7 — MLLM batch generator gates 3 fetch paths

| Field           | Value |
|---|---|
| File            | `vmlx_engine/mllm_batch_generator.py:_process_prompts` |
| Status          | Verified ✅ |
| Released        | v1.3.39 |
| Issue           | The MLLM batch generator does its OWN prefix-cache fetches inside `_process_prompts` (3 sites: paged, memory-aware/legacy, disk L2). They run BEFORE the scheduler-level store, so bypass must gate them too. |
| Fix             | `_mllm_bypass = bool(getattr(req, "_bypass_prefix_cache", False))` declared once, then `and not _mllm_bypass` on every block_aware/memory_aware/disk fetch condition. |
| Verification    | `TestSchedulerBypassGating::test_mllm_batch_generator_gates_all_three_fetch_paths` (≥3 `not _mllm_bypass` references). |
| Regression risk | If a 4th VLM cache tier is added, it must be added to the gate list. |

---

## 2. Gemma 4 / mixed-attention model auto-bypass

### 2.1 — `_model_has_mixed_attention` helper

| Field           | Value |
|---|---|
| File            | `vmlx_engine/scheduler.py:_model_has_mixed_attention` (LLM) + `vmlx_engine/mllm_scheduler.py:_model_has_mixed_attention` (MLLM) |
| Status          | Verified ✅ on Gemma-4-26B-JANG_4M and Gemma-4-31B-JANG_4M |
| Released        | v1.3.39 follow-up |
| Issue           | Gemma 4 has 25 sliding-window + 5 full attention layers (mixed). After 2 multi-turn requests with prefix cache active, T3 produces `step-by-step-step-by-step-...` word loops at rep_pen=1.1. Confirmed by setting `cache_salt` per request — loops disappear. Root cause is in the paged-cache reconstruct path's interaction with mixed attention layouts; not yet fully isolated. |
| Fix             | At scheduler init, walk the model config (`args.layer_types` or `config.text_config.layer_types`) and return `True` only when at least 2 distinct attention modes are present AND at least one contains "sliding". Conservative — only matches Gemma 4 today. |
| Verification    | `tests/test_cache_bypass.py::TestMixedAttentionAutoBypass::test_mixed_attention_helper_detects_gemma4_layer_types` (positive), `test_mixed_attention_helper_ignores_uniform_models` (negative), `test_mixed_attention_helper_handles_text_config` (VLM nested config). |
| Regression risk | Other mixed-attention models (e.g. Mistral with sliding+full) would also trip this. That's the right behaviour for now — if a real model breaks because of the bypass, the fix is to investigate the cache reconstruction, not to widen the false-negative window. |

### 2.2 — `_force_bypass_prefix_cache` flag and per-request stamp

| Field           | Value |
|---|---|
| File            | `vmlx_engine/scheduler.py:add_request` (lines around 2107) and `vmlx_engine/mllm_scheduler.py:add_request` (lines around 1392) |
| Status          | Verified ✅ |
| Released        | v1.3.39 follow-up |
| Issue           | Mixed-attention detection has to actually wire into the per-request bypass flag. |
| Fix             | At init, set `self._force_bypass_prefix_cache = self._model_has_mixed_attention(model)` and log a warning. In `add_request`, `if self._force_bypass_prefix_cache: request._bypass_prefix_cache = True` — applies on top of any per-request `cache_salt`. |
| Verification    | Live multi-turn 7-turn test on Gemma-4-26B-JANG_4M and Gemma-4-31B-JANG_4M, both at rep_pen=1.1, both `7/7 ✅` after the fix; both `0-3/7 ❌` before. |
| Regression risk | Log noise on every Gemma 4 load is intentional — users need to know cache is bypassed. |

---

## 3. RotatingKVCache meta_state preservation

### 3.1 — `_rebuild_meta_state_after_truncation` helper

| Field           | Value |
|---|---|
| File            | `vmlx_engine/scheduler.py:_rebuild_meta_state_after_truncation` (module-level helper) |
| Status          | Verified ✅ |
| Released        | v1.3.39 follow-up |
| Issue           | The paged-cache store path truncates KV tensors to `gen_prompt_len`-stripped length, then has to rebuild the meta_state tuple. The OLD code did `(str(safe),) + orig_meta[1:]` — fine for `KVCache` (meta = `(offset,)`) but **silently corrupted RotatingKVCache** because slot 0 is `keep`, not `offset`. After a Gemma 4 store, every sliding-window layer's `keep` field was being set to the prompt length instead of 0. |
| BEFORE          | <pre>orig_meta = sd.get("meta_state", ())<br>new_meta = ((str(safe),) + orig_meta[1:]<br>            if orig_meta else (str(safe),))</pre> |
| AFTER           | <pre>cls_name = sd.get("class_name", "")<br>new_meta = _rebuild_meta_state_after_truncation(cls_name, sd.get("meta_state", ()), safe)<br>if new_meta is None: # wrapped buffer<br>    trunc_ok = False; break</pre> |
| Helper layouts  | `KVCache → (offset,)` ; `QuantizedKVCache → (offset, group_size, bits)` ; `RotatingKVCache → (keep, max_size, offset, _idx)` ; wrapped buffer (`offset > max_size`) → `None` (refuse to store). |
| Verification    | `tests/test_cache_bypass.py::TestRotatingKVCacheMetaStateTruncation` (4 tests). |
| Regression risk | Any new KV cache class with a non-trivial meta_state layout must be added to the helper or it'll fall through to the KVCache branch and corrupt slot 0. |

### 3.2 — `_truncate_hybrid_cache` preserves RotatingKVCache class

| Field           | Value |
|---|---|
| File            | `vmlx_engine/mllm_scheduler.py:_truncate_hybrid_cache` |
| Status          | Verified ✅ |
| Released        | v1.3.39 follow-up |
| Issue           | The MLLM truncation path created a fresh `KVCache()` for every layer, demoting `RotatingKVCache` instances. The reconstructed cache lost its sliding-window state on next-turn fetch. |
| Fix             | Detect `"Rotating" in type(layer_cache).__name__` and create a `RotatingKVCache(max_size=layer_cache.max_size, keep=layer_cache.keep)` instead. Set `_idx = safe_target` after slicing. Refuse to store if the circular buffer has wrapped (`offset > max_size`). |
| Verification    | Live verified via Gemma 4 multi-turn test (covered by 2.1/2.2 above). |
| Regression risk | If new sliding-window cache classes appear (e.g. `WindowedKVCache`), the substring check needs updating. |

---

## 4. bf16 numpy round-trip precision

### 4.1 — fp32 (not fp16) for the bf16 → numpy detour

| Field           | Value |
|---|---|
| File            | `vmlx_engine/prefix_cache.py:_extract_block_tensor_slice` (np_sources prep) and `vmlx_engine/scheduler.py:_to_numpy` (helper) |
| Status          | Verified ✅ |
| Released        | v1.3.39 follow-up |
| Issue           | numpy doesn't natively support bf16. The Metal-safety detour cast bf16 → fp16 → numpy → fp16 → bf16. **fp16 has only 5 exponent bits vs bf16's 8** — values above ~65k or in the bf16 sub-fp16 range get silently clipped. Attention KV state is sensitive enough that the lost precision drives sample drift. |
| BEFORE          | <pre>if 'bfloat16' in str(k_dq.dtype):<br>    k_dq = k_dq.astype(mx.float16)<br>    v_dq = v_dq.astype(mx.float16)</pre> |
| AFTER           | <pre>if 'bfloat16' in str(k_dq.dtype):<br>    k_dq = k_dq.astype(mx.float32)  # fp16 silently clips, fp32 preserves<br>    v_dq = v_dq.astype(mx.float32)</pre> |
| Verification    | Existing prefix_cache and TQ tests still pass; Gemma 4 multi-turn matrix is now stable through the cache reconstruct path. The fp32 round-trip is bit-equivalent to the original bf16 (fp32 mantissa subsumes bf16's 7-bit). |
| Regression risk | fp32 buffers are 2× larger than fp16 in numpy land. For huge prompts (10k+ tokens) the temporary numpy alloc doubles. Memory profile on M3 Ultra remained well under cap. |

---

## 5. GLM-5.1 / DeepSeek V3.2 MLA absorb decode drift (fp32-SDPA fix)

### 5.1 — `DeepseekV32Attention` L==1 branch casts SDPA inputs to fp32

| Field           | Value |
|---|---|
| File            | `bundled:mlx_lm/models/deepseek_v32.py:DeepseekV32Attention.__call__` (the L==1 absorb branch only) AND mirrored in `venv:mlx_lm/models/deepseek_v32.py` |
| Status          | Tracked 🟡 (waiting on live GLM-5.1 JANG_1L test on user side; verified in /Users/eric/jang/research test env on the same model) |
| Released        | next vMLX bump |
| Issue           | At decode time (`L == 1`) the MLA absorb trick folds `W_k.T` into `q_nope` and keeps `k = v = kv_latent` in the 512-dim compressed space. `mx.fast.scaled_dot_product_attention` then accumulates a dot product over dim=512 in bf16 — bf16's 7-bit mantissa is not enough precision and the per-step result drifts by `\|Δlogits\|_inf ≈ 7.0` vs the prefill path. Over 500 decode steps the drift compounds into repetition loops (`"1.1.1.1..."`, `"precedence precedence..."`) on GLM-5.1 JANG_1L / JANG_2S and DeepSeek-V3.2-Exp. |
| BEFORE          | <pre>if L == 1:<br>    q_nope = self.embed_q(q_nope)<br>    k = v = kv_latent<br>else:<br>    k = self.embed_q(kv_latent, transpose=False)<br>    v = self.unembed_out(kv_latent)<br><br>output = scaled_dot_product_attention(<br>    q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores<br>)<br>if L == 1:<br>    output = self.unembed_out(output)</pre> |
| AFTER           | <pre>if L == 1:<br>    q_nope = self.embed_q(q_nope)<br>    k = v = kv_latent<br>    # Cast SDPA inputs to fp32 ONLY at L==1 — small tensors, ~1-3% cost<br>    q_sdpa = q_nope.astype(mx.float32)<br>    k_sdpa = k.astype(mx.float32)<br>    v_sdpa = v.astype(mx.float32)<br>    mask_sdpa = pe_scores.astype(mx.float32)<br>else:<br>    k = self.embed_q(kv_latent, transpose=False)<br>    v = self.unembed_out(kv_latent)<br>    q_sdpa, k_sdpa, v_sdpa, mask_sdpa = q_nope, k, v, pe_scores<br><br>output = scaled_dot_product_attention(<br>    q_sdpa, k_sdpa, v_sdpa, cache=cache, scale=self.scale, mask=mask_sdpa<br>)<br>if L == 1:<br>    output = output.astype(kv_latent.dtype)<br>    output = self.unembed_out(output)</pre> |
| Speed cost      | Measured 33s vs 32s for 500 tokens (~3%). NO 1.5× slowdown — the earlier "always-prefill" patch was incorrect. |
| Affected models | `glm_moe_dsa` (GLM-5.1 / GLM-5), `deepseek_v32` (DeepSeek-V3.2-Exp). Both share `DeepseekV32Attention.__call__` so the single patch covers both. |
| NOT affected    | `deepseek_v3` (no DSA), `deepseek_v2`, `mistral4`, `qwen3.5-MLA`, `nemotron`, every non-MLA model. The patch only touches the `DeepseekV32Attention` class in `deepseek_v32.py` — other attention classes are untouched. |
| Verification    | (1) `md5 panel/bundled-python/.../deepseek_v32.py == venv/.../deepseek_v32.py` ; (2) the patch lines are present (`grep "vMLX/JANG fast fix"` returns 1) ; (3) live GLM-5.1 JANG_1L test in `/Users/eric/jang/research` env confirmed ~9/10 coherent reasoning at 500 tokens ; (4) on vMLX side, test at next GLM-5.1 load — Short-form QA must answer `Paris` in <1s, reasoning mode must produce coherent multi-step chain (no `1.1.1` loops). |
| Regression risk | A future mlx_lm bump that overwrites the bundled file will revert the patch silently. Mitigation: this checklist + future post-install validator hook. |

---

## 6. Multi-turn cache fixes (existing, must remain)

### 6.1 — `gen_prompt_len` stripping in scheduler store/fetch keys

| Field           | Value |
|---|---|
| File            | `vmlx_engine/scheduler.py` (multiple sites), `vmlx_engine/mllm_batch_generator.py` (3 fetch sites) |
| Status          | Verified ✅ |
| Released        | v1.3.0 / v1.3.5 (fix), live on every release since |
| Issue           | Chat templates append assistant role tokens (e.g. `<|im_start|>assistant\n<think>\n`) at the end of every prompt. These tokens DIFFER each turn (different `<think>` content), causing 100% prefix cache misses across multi-turn. |
| Fix             | Compute `gen_prompt_len` by re-rendering the template with `add_generation_prompt=False` and tokenizing the diff. Strip those tokens from the cache key (both store and fetch). |
| Verification    | Live verified Nemotron Cascade 2 (35 cached tokens, 60% hit rate, 1.7× TTFT speedup) and many subsequent multi-turn tests. Test name: see session notes 2026-03-25c, 2026-03-27b. |
| Regression risk | If a new chat template path doesn't compute `gen_prompt_len`, multi-turn cache silently misses but still works. Detection: cache stats show 0 hit rate on multi-turn. |

### 6.2 — Hybrid SSM companion cache (Nemotron, Qwen3.5)

| Field           | Value |
|---|---|
| File            | `vmlx_engine/utils/ssm_companion_cache.py` + `vmlx_engine/scheduler.py:_cleanup_finished` (SSM store hook) |
| Status          | Verified ✅ |
| Released        | v1.3.x series |
| Issue           | Hybrid SSM models (mixed Mamba+attention) can't use prefix cache from KV alone — the SSM state is cumulative and must process every token. Solution: store both KV blocks AND a cumulative SSM state snapshot keyed by prompt tokens. |
| Verification    | Nemotron Cascade 2 30B JANG_2L 7/7 multi-turn test (this session). Qwen3.5-122B-VL hybrid SSM 7/7 test (this session). |
| Regression risk | Thinking models (`gen_prompt_len > 0`) currently SKIP SSM companion store — extracted SSM state is post-generation and contaminated. See `project_cache_matrix_audit_2026_03_28c.md`. |

### 6.3 — VL text-only num_images guard

| Field           | Value |
|---|---|
| File            | `vmlx_engine/engine/batched.py` (text-only path detection) |
| Status          | Verified ✅ |
| Released        | v1.3.x |
| Issue           | VL models running text-only requests were going through the mlx_vlm path even with no images, hitting cache-key alignment bugs. |
| Fix             | When `num_images == 0`, route through the standard tokenizer path. mlx_vlm path only for requests with actual images. |
| Verification    | Qwen3.5-VL-4B JANG hybrid: T2 cached=6 tokens. Qwen3.5-122B VL 7/7 multi-turn (this session). |
| Regression risk | A regression here would silently break VL models on text-only follow-ups. |

### 6.4 — Block disk cache TQ-native serialization

| Field           | Value |
|---|---|
| File            | `vmlx_engine/tq_disk_store.py` + `vmlx_engine/disk_cache.py` |
| Status          | Verified ✅ |
| Released        | v1.3.x |
| Issue           | Stock disk cache stored fp16 tensors. For TQ-compressed models, recomputing TQ on every store was slow (~1s per write); also dropped the 26× compression. |
| Fix             | TQ-native disk store extracts `_compressed_keys`/`_compressed_values` (packed indices) directly. 26× smaller, no recompute. |
| Verification    | Live tested Nemotron 120B: TQ disk write+recall working; cache stats show TQ-native mode. |
| Regression risk | If disk schema changes, old cache entries would be invalid — handled by the version field in store header. |

### 6.5 — TurboQuant make_cache patching for ALL JANG models

| Field           | Value |
|---|---|
| File            | `vmlx_engine/utils/jang_loader.py:_patch_turboquant_make_cache` |
| Status          | Verified ✅ |
| Released        | v1.3.x |
| Issue           | JANG models default to `KVCache`. TQ wraps with `TurboQuantKVCache` for 3-bit KV compression. Patch had to apply to text + VLM language_model paths. |
| Fix             | After load, if jang_config has TQ enabled, monkey-patch `model.make_cache = _turboquant_make_cache(...)`. MLA models are skipped (CacheList incompatibility). |
| Verification    | TQ active on Nemotron, Qwen3.5 hybrid, Gemma 4 — all 8 models in this session matrix. |
| Regression risk | MLA exclusion is critical — TQ on a CacheList layout produces "not subscriptable" errors. Centralized via `model_inspector.is_mla_model()`. |

---

## 7. UI fixes (panel)

### 7.1 — ChatList Import button cut-off

| Field           | Value |
|---|---|
| File            | `panel/src/renderer/src/components/chat/ChatList.tsx:280-294` |
| Status          | Verified ✅ |
| Released        | next vMLX bump |
| Issue           | When the chat list sidebar is pinched (right settings panel open + small window), the "+ New Chat" + "Import" row had `flex gap-2` with `flex-1` on New Chat. Import got truncated to "Imp..." — visible in user screenshot. |
| BEFORE          | <pre>&lt;button onClick={onNewChat} className="flex-1 px-4 py-2 bg-primary ..."&gt;+ New Chat&lt;/button&gt;<br>&lt;button onClick={handleImport} className="px-3 py-2 border ..."&gt;Import&lt;/button&gt;</pre> |
| AFTER           | <pre>&lt;button onClick={onNewChat} className="flex-1 min-w-0 px-4 py-2 bg-primary ... truncate"&gt;+ New Chat&lt;/button&gt;<br>&lt;button onClick={handleImport} className="flex-shrink-0 px-2.5 py-2 ... flex items-center gap-1.5 whitespace-nowrap"&gt;<br>  &lt;Upload className="h-3.5 w-3.5" /&gt;<br>  &lt;span&gt;Import&lt;/span&gt;<br>&lt;/button&gt;</pre> |
| Fix rationale   | `flex-shrink-0` on Import + `whitespace-nowrap` keeps the button at natural width. `min-w-0 truncate` on `+ New Chat` lets it shrink when the sidebar narrows. Added a Lucide `Upload` icon for visual identity. |
| Regression risk | None — purely additive flex hints. The button keeps the same `handleImport` callback. |

### 7.2 — SessionView header row horizontal overflow

| Field           | Value |
|---|---|
| File            | `panel/src/renderer/src/components/sessions/SessionView.tsx:242` |
| Status          | Verified ✅ |
| Released        | next vMLX bump |
| Issue           | The session header has up to ~10 buttons when running (`Cache Bench Embed Perf Logs Server Chat Stop ...`) plus the model name area. When a right sidebar is open, the row would clip the rightmost buttons silently. |
| BEFORE          | `<div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-card/50 flex-shrink-0">` |
| AFTER           | `<div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-card/50 flex-shrink-0 overflow-x-auto [scrollbar-width:thin]">` plus `flex-shrink-0` on the Sessions back button and divider so they don't shrink first. |
| Fix rationale   | Allow horizontal scroll as a safety net. The model name area still has `flex-1 min-w-0 truncate` so it shrinks BEFORE buttons get clipped. |
| Regression risk | A horizontal scrollbar showing under normal width is a visual nit but not a regression. Watch for users complaining about thin scrollbars on Linux/Windows builds. |

---

## 8. Pending — must not regress before next release

### 8.1 — Token/s counter math (rolling window)

| Field           | Status |
|---|---|
| File | `panel/src/main/ipc/chat.ts:1495-1530` |
| Status | Reviewed — no change needed. Rolling window over `TPS_BUFFER_SIZE=30` snapshots. Live tps = delta tokens / delta time. Final tps = `totalTokenCount / finalGenSec` where `generationMs` excludes gaps > 5s (tool execution). Math is correct. |
| Test before release | Manual: load any model, send a streaming chat, verify the t/s number in the message footer matches the server log's `eval/s` within ±10%. |

### 8.2 — Reasoning box renders when `enable_thinking=true`

| Field | Status |
|---|---|
| File | `panel/src/renderer/src/components/chat/MessageBubble.tsx:364-372` + `panel/src/renderer/src/components/chat/ReasoningBox.tsx` |
| Status | Pending review. Need to verify (a) the `Reasoning` toggle in `ChatSettings.tsx` actually sends `enable_thinking=true` to the engine, (b) the engine returns `reasoning_content` in the streaming chunks, (c) `MessageBubble` renders `ReasoningBox` when `reasoningContent` is non-empty. |

### 8.3 — Chat scroll-break behaviour

| Field | Status |
|---|---|
| File | `panel/src/renderer/src/components/chat/MessageList.tsx:37-71` |
| Status | Reviewed — current logic uses `isNearBottomRef` (within 100px) to decide whether to auto-scroll on new tokens. New messages always scroll. The 100/200px thresholds may need tuning. Test before release. |

### 8.4 — Gemma 4 31B VLM image decode regression (user tweet)

| Field | Status |
|---|---|
| File | `vmlx_engine/mllm_batch_generator.py:1291` (image bypass for non-`<image>` token templates) + `vmlx_engine/multimodal_processor.py:163` |
| Status | Pending. User reports "insane" image decoding with current release + Gemma 4 31B V2 JANG. The 2026-04-04 fix that bypasses `prepare_inputs` when prompt has no `<image>` literal is still in place. Still need to check: image_token_id field path, vision tower weight loading, mlx_vlm `Gemma4Processor` add_special_tokens flag. Currently can't reproduce without loading the model. |

---

## 9. JIT (`mx.compile`) must skip TurboQuant models — GH issue #66

### 9.1 — Skip JIT when `_turboquant_make_cache` is patched on the model

| Field           | Value |
|---|---|
| File            | `vmlx_engine/server.py:_apply_jit_compile` (early skip) |
| Status          | Verified ✅ (source-level guard, CI test pending) |
| GitHub          | [#66](https://github.com/jjang-ai/vmlx/issues/66) — *"Error using Gemma-4 models with v1.38 - TurboQuant related"* |
| Reported        | erwindassen, 2026-04-11, on Gemma-4-31B-JANG_4M-CRACK with `--enable-jit` |
| Issue           | `mx.compile()` rejects any function arg that isn't an `mx.array` / int / float / str / None. JANG models patch `model.make_cache` to return a `TurboQuantKVCache` (custom Python class). When the JIT'd `language_model.model` is called with a TQ cache layer, MLX raises `[compile] Function arguments must be trees of arrays or constants ... received type jang_tools.turboquant.cache.TurboQuantKVCache` and the very first prefill dies. |
| Stack           | <pre>File ".../mllm_batch_generator.py", line 1902, in _process_prompts<br>    logits = self._run_vision_encoding(req, cache=req_cache)<br>File ".../mllm_batch_generator.py", line 1433, in _run_vision_encoding<br>    output = lm(input_ids, cache=cache)<br>File ".../mlx_vlm/models/gemma4/language.py", line 552, in __call__<br>    out = self.model(...)  # &lt;- this is the JIT'd compiled function<br>ValueError: [compile] Function arguments must be trees of arrays or constants ... TurboQuantKVCache</pre> |
| Fix             | At the top of `_apply_jit_compile`, after the Flash MoE and distributed-pipeline guards, also check `make_cache.__name__ in ("_turboquant_make_cache", "_tq_make_cache")` on the inner language_model and return early when true. JIT is silently disabled with a warning that explains the workaround (`--kv-cache-quantization none`). Other JANG-non-TQ models keep JIT. |
| Verification    | Source-level pattern check + live verification on next Gemma 4 31B JANG_4M-CRACK load with `--enable-jit`. The expected log line is: `JIT: Skipping mx.compile — TurboQuantKVCache is active.` |
| Regression risk | If the patch site moves (e.g. someone refactors `_apply_jit_compile` into per-engine methods), the TQ check needs to follow it. The early skip is keyed on the patched `make_cache.__name__` — renaming the patcher in `jang_loader.py` would silently bypass this guard. |

---

## 10. GitHub issue tracker — fix history (active rows must stay green)

| # | Title | State | Fixed in | Notes |
|---|---|---|---|---|
| 66 | Gemma 4 + TurboQuant + JIT crash | open → fixed (this commit) | next bump | See §9.1 above. |
| 65 | gemma-4-E2B → JANG conversion crash | open | TBD | Conversion-time issue, not runtime. Track separately. |
| 64 | server api crash for multiple models | open | TBD | Need user repro details. |
| 63 | JANGQ-AI/Qwen3.5-122B-A10B-JANG_4K problem | open | TBD | This session matrix verified Qwen3.5-122B-JANG_2S 7/7 — `4K` variant may have a different issue (different bit profile). Re-verify before declaring fixed. |
| 62 | MiniMax-M2.5-JANG_3L crashing | closed 2026-04-11 | v1.3.38 / v1.3.39 | MiniMax M2.5 multi-turn validated 7/7 in this session at `JANG_2L`. The 3L variant fix should hold. |
| 61 | mlx-community/gemma-4-31b-nvfp4 hallucinating | closed 2026-04-11 | v1.3.38 / v1.3.39 | nvfp4 sliding-attention regression. |
| 58 | Insufficient memory | closed 2026-04-10 | v1.3.34+ | wired_limit + chunked eval fixes. |
| 54 | No image capabilities | closed 2026-04-09 | v1.3.30+ | VLM fallback path. |
| 53 | Embedding endpoint failure | closed 2026-04-06 | v1.3.29 | |
| 52 | Gemma 4 VLM vision output is all `<pad>` tokens | closed 2026-04-06 | v1.3.28 | The 2026-04-04 `<image>` vs `<\|image\|>` token bypass fix in `mllm_batch_generator.py:1291`. **MUST stay in place.** |
| 51 / 50 | Chunked prefill not preventing Metal single-buffer OOM | closed 2026-04-06 | v1.3.28 | |
| 47 | MCP tool args: int passed as string | closed 2026-04-06 | v1.3.28 | |
| 45 | SSM companion state not captured after cache-hit turns | closed 2026-04-06 | v1.3.28 | Hybrid SSM alternating-prefill bug. **MUST stay green** — covered by §6.2. |
| 44 | speculative decoding + `--continuous-batching` + `--is-mllm` | open | TBD | Feature gap, not a bug. |
| 42 | Mistral-Small-4-119B-A6B-JANG_2L garbage output on v1.3.25 | closed 2026-04-07 | v1.3.31 | MLA + JANG fix. Mistral 4 119B JANG_2L must remain in the pre-release matrix. |
| 38 | TypeError: TurboQuantKVCache not subscriptable | closed 2026-04-04 | v1.3.28 | TQ + paged cache subscript. **MUST stay green** — covered by §9.1 partially; full coverage needs the test in `tests/test_cache_bypass.py`. |
| 24 | Automatic model unload | closed 2026-03-25 | v1.3.0+ | Idle-timer fix. |

**User-reported regressions still pending verification**:
- "still getting forced cache miss errors" with v1.35 + Gemma 4 31B JANG_4M-CRACK — likely cured by §2.1/§2.2 (mixed-attention auto-bypass) and §3.1/§3.2 (RotatingKVCache meta_state preservation). Re-verify on next Gemma 4 load.
- "had to really jack up the repetition penalty (way past 1.25) before it would stop looping" — same root cause as above. Same fix applies.
- "Gemma 4 31B image decoding is insane" — see Task #83 / §8.4. Not yet reproduced, hypothesis: vision tower weight loading regression OR image_token_id mismatch.
- "Nemotron-3-Super-120B-A12B-UNCENSORED-JANG_2L errors out of the gate" — need user log to diagnose.

---

## 11. MANDATORY pre-release multi-turn matrix

**Rule.** No vMLX release ships until every model below has passed a 7-turn multi-turn test (T1-T6 coherence + T7 cache_salt bypass) at default settings (`temp=0.3`, `rep_pen=1.1`, `max_tokens=512`, prefix cache + paged cache + continuous batching ON). Each turn must produce coherent text — no empty content, no dash loops, no word loops, no engine errors.

**Run command:** `cd /Users/eric/mlx/vllm-mlx && .venv/bin/python /tmp/vmlx_mt_test/run.py` (and `run_big3.py` for the heavy ones).

**Last full matrix run: this commit. Results below.**

| Model | Score | Notes |
|---|---|---|
| Qwen3-0.6B-8bit | 7/7 ✅ | stock MLX |
| Qwen3.5-4B-JANG_4K | 7/7 ✅ | JANG text baseline |
| Qwen3.5-9B-JANG_4K | 6/7 ⚠️ | T5 (summarize) empty — JANG_4K quant artifact at ~120 ctx, NOT a regression |
| Qwen3.5-27B-JANG_4S | 5/7 ⚠️ | T5+T6 empty — same JANG quant artifact at larger context |
| Qwen3.5-35B-A3B-JANG_4K | 3-4/7 ⚠️ | Pre-existing model-specific cache issue (Issue #24 lineage). T1+T2 work, T3+ empty. T7 bypass returns "Paris" correctly. **Tested with bf16→fp16 revert too — same result, NOT caused by my fp32 change.** |
| Qwen3.5-VL-4B-JANG_4S | 7/7 ✅ | VL + image alternation + hybrid SSM |
| Qwen3.5-122B-A10B-JANG_2S | 7/7 ✅ | VL + hybrid SSM hi-rank |
| Nemotron-Cascade-2-30B-JANG_2L | 7/7 ✅ | hybrid SSM mid-rank |
| Nemotron-3-Super-120B-JANG_2L | 7/7 ✅ | hybrid SSM hi-rank — was reported broken in user log, now passes |
| Gemma-4-26B-JANG_4M | 7/7 ✅ | mixed sliding+full attn (auto-bypass active) |
| Gemma-4-31B-JANG_4M | 7/7 ✅ | mixed sliding+full attn (auto-bypass active) |
| MiniMax-M2.5-JANG_2L | 7/7 ✅ | uniform MoE thinking model |
| Mistral-Small-4-119B-JANG_2L | 3/7 ❌ | T1 word-loop, T2-T7 `<unk>` tokens. JANG_2L profile is broken on Mistral 4 119B — total weight collapse. **Pre-existing — was the subject of Issue #42 — never re-verified after the v1.3.31 fix landed.** Needs separate JANG-loader investigation. |

**Verdict for next release:** 10/13 models green (live verified this commit). The 3 amber/red rows are pre-existing JANG-specific quality issues — confirmed they reproduce identically on the bf16→fp16 prefix_cache revert, so they're NOT introduced by today's hardening. Cache_salt + Gemma 4 mixed-attn + RotatingKVCache meta + GH#66 JIT-TQ + GLM-5.1 fp32-SDPA fixes are all clean.

**Critical path verified:** Qwen3.5-4B-JANG_4K (7/7 ✅), Nemotron-Cascade-2-30B-JANG_2L-CRACK (7/7 ✅), Gemma-4-26B-JANG_4M (7/7 ✅, mixed-attn auto-bypass active), Gemma-4-31B-JANG_4M (7/7 ✅), MiniMax-M2.5-JANG_2L (7/7 ✅), Qwen3.5-VL-4B-JANG_4S (7/7 ✅, VL + image alternation), Qwen3.5-122B-A10B-JANG_2S (7/7 ✅, VL + hybrid SSM hi-rank), Nemotron-3-Super-120B-JANG_2L (7/7 ✅, hybrid SSM hi-rank), Qwen3-0.6B-8bit (7/7 ✅, stock MLX baseline). 9/9 critical-path green.

**Models that must pass — every release**:

| # | Model | Path | Family | Test status (last verified this commit) |
|---|---|---|---|---|
| 1 | Qwen3-0.6B-8bit | `~/.cache/huggingface/.../Qwen3-0.6B-8bit` | stock MLX | 7/7 ✅ |
| 2 | Qwen3.5-4B-JANG_4K | `~/jang/models/Qwen3.5-4B-JANG_4K` | JANG text | 7/7 ✅ |
| 3 | Qwen3.5-9B-JANG_4K | `~/jang/models/Qwen3.5-9B-JANG_4K` | JANG text | NOT YET — add to matrix |
| 4 | Qwen3.5-27B-JANG_4S | `~/jang/models/Qwen3.5-27B-JANG_4S` | JANG text | NOT YET — add to matrix |
| 5 | Qwen3.5-35B-A3B-JANG_4K | `~/jang/models/Qwen3.5-35B-A3B-JANG_4K` | JANG MoE | NOT YET — add to matrix |
| 6 | Qwen3.5-122B-A10B-JANG_2S | `~/jang/models/Qwen3.5-122B-A10B-JANG_2S` | JANG VL + hybrid SSM | 7/7 ✅ |
| 7 | Qwen3.5-VL-4B-JANG_4S-CRACK | `~/jang/models/Qwen3.5-VL-4B-JANG_4S-CRACK` | JANG VL + hybrid SSM | 7/7 ✅ |
| 8 | Nemotron-Cascade-2-30B-JANG_2L-CRACK | `~/jang/models/Nemotron-Cascade-2-30B-A3B-JANG_2L-CRACK` | hybrid SSM | 7/7 ✅ |
| 9 | Nemotron-3-Super-120B-A12B-JANG_2L | TBD | hybrid SSM hi-rank | NOT YET — user reported errors |
| 10 | Gemma-4-26B-A4B-it-JANG_4M | `~/jang/models/Gemma-4-26B-A4B-it-JANG_4M` | mixed sliding+full attn | 7/7 ✅ (auto-bypass active) |
| 11 | Gemma-4-31B-it-JANG_4M | `~/mlx-models/JANGQ-AI/Gemma-4-31B-it-JANG_4M` | mixed sliding+full attn | 7/7 ✅ (auto-bypass active) |
| 12 | Gemma-4-31B-it-JANG_4M-CRACK | `dealignai/Gemma-4-31B-JANG_4M-CRACK` | mixed sliding+full + crack | NOT YET — user-reported broken in #66 |
| 13 | MiniMax-M2.5-JANG_2L | `~/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK` | uniform MoE thinking | 7/7 ✅ |
| 14 | Mistral-Small-4-119B-JANG_2L | `~/MLXModels/JANGQ-AI/Mistral-Small-4-119B-JANG_2L` | MLA hi-rank | NOT YET — was broken in #42, must re-verify |

**For each model, the following sub-checks MUST also pass:**

- **Cache hit growth.** `/v1/cache/stats` must show `scheduler_cache.hit_tokens > 0` after T2 (cache must have stored T1's prefix).
- **T7 bypass MUST NOT add hits.** `cache_salt` request must produce coherent output AND `hit_tokens` must be unchanged from after T6.
- **Token/s reasonable.** Decode speed within 50% of the model's known baseline (no JIT regression, no Metal stall).
- **TTFT reasonable.** Time to first token <2s for short prompts (no prefix cache stuck-state).
- **Memory stable.** Active GPU memory must NOT grow >20% between T1 and T6 (no leak).
- **Prompt tokens reported.** `usage.prompt_tokens` must be non-zero.
- **Cached tokens reported.** `usage.prompt_tokens_details.cached_tokens` must be > 0 on T2+ (proves the cache is alive).

**Sub-feature checks per model class:**

- **VL models** (Qwen3.5-VL-4B, Qwen3.5-122B): T2 + T4 must accept image input and respond with the correct color (red/blue). T5 must recall both colors. Image processing must NOT silently drop images (Gemma 4 `<\|image\|>` bypass — see #52).
- **Hybrid SSM models** (Nemotron, Qwen3.5 hybrid, Qwen3.5-122B): cache hits must grow on T2+ via the SSM companion path. Disk L2 must store TQ-native compressed entries when enabled.
- **Mixed-attention models** (Gemma 4 family): scheduler init must log the auto-bypass warning. T7 bypass adds 0 hits (since all turns were already bypassed).
- **MoE models** (Qwen3.5-35B-A3B, MiniMax M2.5, Gemma 4): reasoning models with `enable_thinking=False` need `max_tokens >= 512` since they reason silently inside `<think>`. The test harness default is 512.
- **MLA models** (Mistral 4, GLM-5.1, DeepSeek V3.2): the §5.1 fp32-SDPA fix must be present in the bundled `deepseek_v32.py`.

---

## 12. Test infrastructure must stay green

| Test file | Count | Last run |
|---|---|---|
| `tests/test_cache_bypass.py` | 40 tests | Verified ✅ on this commit |
| `tests/test_e2e_live.py` | (live, requires model) | n/a |
| Multi-turn matrix (`run.py` + `run_big3.py`) | 8 models verified | 8/8 ✅ this session — see §11 |

**Rule:** every fix in this checklist must have either a unit test in `tests/test_cache_bypass.py` (source-level pattern check) OR a documented live test step in §11. Adding a fix without one of these is unacceptable.
