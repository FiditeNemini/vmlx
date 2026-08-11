# MiniMax-M3 EAGLE3 Native MTP Handoff

Date: 2026-06-13
Machine: erics-m5-max.local
Runtime checkout: /Users/eric/mlx/vllm-mlx
Branch: feat/minimax-m3-runtime

## Current State

The REAP22 MiniMax-M3 bundle now carries a self-contained EAGLE3 sidecar:

- Bundle: /Users/eric/.mlxstudio/models/JANGQ-AI/MiniMax-M3-REAP22-JANG_2L
- Sidecar config: eagle3_config.json
- Sidecar weights: eagle3_runtime.safetensors
- Method exposed to vMLX: native_mtp with native_mtp_method=eagle3

This is not config-declared MiniMax MTP. The public MiniMax M3 MTP tensors are absent from
the target index. Treat the EAGLE3 sidecar as the native-MTP implementation for this bundle.

## Implemented In Tree

- vmlx_engine/native_mtp.py
  - Detects eagle3_config.json before falling back to mtp.* index detection.
  - Reports method=eagle3, adapter=minimax_m3_eagle3, tap layers, weights file, identity token map,
    and measured tokens_per_target_forward.
  - Reports runtime_available=false until the verify loop is wired.

- vmlx_engine/models/minimax_m3/eagle3_draft.py
  - Drop-in EAGLE3 draft head loader and KV-cached draft recurrence.
  - load_eagle3_draft(bundle) loads the sidecar config + weights.

- vmlx_engine/models/minimax_m3/minimax_m3.py
  - Adds return_aux=True and aux_layers=(2,30,57) support.
  - Captures DecoderLayer output residuals for EAGLE3 target taps.
  - Default return_aux=False path is unchanged.

- vmlx_engine/models/minimax_m3/cache.py
  - Adds truncate_minimax_m3_cache(cache, length).
  - Uses each layer cache trim implementation, so sparse MSA layers keep K/V and idx_keys aligned.

- tests/test_native_mtp_autodetect.py
  - Adds MiniMax-M3 EAGLE3 sidecar native-MTP candidate coverage.

- vmlx_engine/models/minimax_m3/test_m3_eagle3_native_mtp.py
  - Verifies aux taps and cache truncation scaffolding on a tiny MiniMax-M3 model.

## Verified

Commands run on erics-m5-max.local:

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest tests/test_native_mtp_autodetect.py -k "minimax_m3_eagle3" -q
cd vmlx_engine/models/minimax_m3
.venv/bin/python -m py_compile minimax_m3.py cache.py eagle3_draft.py test_m3_eagle3_native_mtp.py
.venv/bin/python test_m3_eagle3_native_mtp.py
.venv/bin/python test_m3_affine2_switch.py
.venv/bin/python test_m3_tq_skip.py
```

Real bundle status:

```text
status weights_present_runtime_unwired
family minimax_m3
native_mtp_method eagle3
runtime_adapter minimax_m3_eagle3
artifact_available True
runtime_supported True
runtime_available False
runtime_mtp_mode eagle3_sidecar
eagle3_weights_file eagle3_runtime.safetensors
eagle3_aux_hidden_state_layers [2, 30, 57]
eagle3_tokens_per_target_forward 1.76
issues []
```

Draft loader smoke:

```text
loaded eagle3 eagle3_runtime.safetensors [2, 30, 57] weights 17
combine (1, 1, 6144) mlx.core.bfloat16
```

## Remaining Work

1. Wire the live MiniMax-M3 decode loop to request return_aux=True on target forwards.
2. Seed EAGLE3 with concat(aux@2, aux@30, aux@57) from the last verified target position.
3. Draft K tokens with Eagle3Draft.propose or the lower-level cached step loop.
4. Verify the drafted chain in one target forward.
5. Accept the longest greedy-matching prefix plus the target bonus token.
6. Roll target cache back with truncate_minimax_m3_cache(cache, accepted_length).
7. Reset/truncate the EAGLE3 draft KV independently.
8. Gate correctness with spec-decode output == plain greedy output, token-for-token.

Do not mark runtime_available=true until step 8 passes on the real bundle.

## Practical Defaults

- Start with K=4.
- Greedy only first: temp=0, repetition_penalty=1.
- Continuous batching should stay off for this path until rollback and per-request draft state are explicit.
- Keep PR #205 speed checkpoint separate unless this full verify loop passes.
