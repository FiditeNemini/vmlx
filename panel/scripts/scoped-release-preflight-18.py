#!/usr/bin/env python3
"""Fail-closed preflight for the vMLX 1.6.18 production checkpoint.

Private release evidence must stay outside the public repository. This gate
consumes a sanitized external attestation, binds every required check to the
exact source commit/tree, and writes only check names and SHA-256 evidence
digests into its public-build manifest. Signing, notarization, installed-app
smoke, tagging, and publication remain downstream gates.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import http.client
import importlib.util
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import stat
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "r18_production"
VERSION = "1.6.18"
SCHEMA = "vmlx-r18-release-attestation-v1"
PROOF_SCHEMA = "vmlx-r18-check-proof-v1"
SOURCE_TRACE_SCHEMA = "vmlx-r18-source-trace-v1"
COMMAND_RESULT_SCHEMA = "vmlx-r18-command-result-v1"
JANG_RESULT_SCHEMA = "vmlx-r18-jang-runtime-result-v1"
ELECTRON_PROOF_SCHEMA = "vmlx-electron-ui-proof-v2"
API_MATRIX_SCHEMA = "vmlx-agentic-protocol-matrix-v2"
CACHE_PROOF_SCHEMA = "vmlx-cache-hierarchy-live-gate-v2"
ELECTRON_RAW_SCHEMA = "vmlx-electron-ui-proof-v4"
API_RAW_SCHEMA = "vmlx-agentic-protocol-matrix-v4"
CACHE_RAW_SCHEMA = "vmlx-cache-hierarchy-live-gate-v4"
PRIVATE_CACHE_ATTESTATION_TOKEN_ENV = "VMLINUX_PRIVATE_CACHE_ATTESTATION_TOKEN"
PRIVATE_CACHE_ATTESTATION_TOKEN_FILE_ENV = (
    "VMLINUX_PRIVATE_CACHE_ATTESTATION_TOKEN_FILE"
)
OWNED_EXECUTION_SCHEMA = "vmlx-r18-owned-execution-v1"
API_CAPTURE_LAYER = "requests.decompressed_response_parser_input"
API_CAPTURE_SEMANTICS = (
    "Exact decompressed response-body bytes delivered to protocol parsers: "
    "streaming bytes before requests.iter_lines line splitting or Unicode "
    "decoding, and nonstream response bytes before JSON decoding; excludes "
    "HTTP transfer framing and compressed transport octets."
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_MARKER_RE = re.compile(
    r"(?is)"
    r"<\s*/?\s*(?:think|thinking|reasoning|tool_call|tool_calls|function|invoke)\b"
    r"|\[(?:THINK|TOOL|TOOL_CALLS?)\]"
    r"|<\|(?:tool_call|tool_calls|point|box)[^>]*\|>"
)
JANG_VERSION = "2.5.34"
JANG_COMMIT = "f7583e445e8bf75abe49b5cd0a19370a1d5ceb41"
JANG_TREE = "00dee8c330fd69697cce530ea667ed6ffed87a97"

BUNDLE_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "jang_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

OWNED_CHECK_NAMES = (
    "full_python_suite",
    "full_panel_suite",
    "typecheck",
    "production_build",
    "jang_runtime_provenance",
)
OWNED_SUITE_BASELINES = {
    "full_python_suite": {"passed": 1_000, "collected": 1_000},
    "full_panel_suite": {"passed": 50, "files": 5},
}

COMMON_FAMILY_ASSERTIONS = (
    "same_source_ui_api_session",
    "electron_minimum_three_turns",
    "raw_api_minimum_three_turns",
    "chat_completions_checked",
    "responses_checked",
    "anthropic_checked",
    "ollama_checked",
    "reasoning_parser_checked",
    "tool_loop_checked",
    "paged_off_ssd_partial_checked",
    "paged_on_l2_refault_checked",
    "model_defaults_checked",
    "rendering_checked",
    "output_coherence_checked",
)

REQUIRED_ASSERTIONS = {
    "exact_source_provenance": (
        "checkout_head_exact",
        "origin_branch_pushed",
        "origin_main_exact",
        "electron_revision_exact",
        "renderer_revision_exact",
        "python_import_revision_exact",
        "engine_pid_recorded",
        "bundle_config_hashes_recorded",
    ),
    "full_python_suite": (
        "complete_suite",
        "focused_suite_not_substituted",
        "terminal_summary_passed",
    ),
    "full_panel_suite": (
        "complete_suite",
        "focused_suite_not_substituted",
        "terminal_summary_passed",
    ),
    "typecheck": ("terminal_summary_passed",),
    "production_build": (
        "exact_checkout_built",
        "terminal_summary_passed",
    ),
    "electron_visual_multiturn": (
        "real_start_button",
        "minimum_three_turns",
        "reasoning_rail_checked",
        "visible_content_checked",
        "tool_result_continuation_checked",
        "terminal_state_checked",
        "cache_ttft_tps_checked",
        "rendering_checked",
        "coherence_checked",
    ),
    "raw_api_chat": (
        "direct_stream_checked",
        "gateway_stream_checked",
        "reasoning_content_separate",
        "content_progressive",
        "tool_result_continuation_checked",
        "history_checked",
        "request_kwargs_checked",
        "terminal_truthful",
    ),
    "raw_api_responses": (
        "direct_stream_checked",
        "gateway_stream_checked",
        "reasoning_content_separate",
        "content_progressive",
        "tool_result_continuation_checked",
        "history_checked",
        "request_kwargs_checked",
        "terminal_truthful",
    ),
    "raw_api_anthropic": (
        "direct_stream_checked",
        "gateway_stream_checked",
        "reasoning_content_separate",
        "content_progressive",
        "tool_result_continuation_checked",
        "history_checked",
        "request_kwargs_checked",
        "terminal_truthful",
    ),
    "raw_api_ollama": (
        "direct_stream_checked",
        "gateway_stream_checked",
        "reasoning_content_separate",
        "content_progressive",
        "tool_result_continuation_checked",
        "history_checked",
        "request_kwargs_checked",
        "terminal_truthful",
    ),
    "interleaved_reasoning_tools": (
        "reasoning_tool_reasoning_tool_answer",
        "exact_tool_arguments",
        "no_control_markup_leak",
        "nonempty_final_answer",
    ),
    "output_integrity_terminal_stats": (
        "no_reasoning_only_finalization",
        "no_stale_reasoning_replay",
        "no_looping_or_gibberish",
        "no_eos_or_truncation_regression",
        "ttft_tps_compared_to_raw_timing",
        "terminal_event_truthful",
    ),
    "cache_paged_off_ssd_partial": (
        "paged_ram_disabled",
        "ssd_l2_enabled",
        "longest_prefix_partial_block_hit",
        "uncached_suffix_prefilled",
        "prefill_skip_measured",
        "cross_chat_reuse",
        "cross_session_reuse",
    ),
    "cache_paged_on_eviction_refault": (
        "paged_ram_enabled",
        "ram_blocks_filled",
        "oldest_unused_evicted",
        "disk_refault_observed",
        "longest_prefix_partial_block_hit",
        "uncached_suffix_prefilled",
    ),
    "cache_restart_and_size_eviction": (
        "restart_disk_restore",
        "disk_size_limit_enforced",
        "disk_oldest_unused_evicted",
        "ram_percentage_limit_enforced",
        "ram_oom_warning_checked",
    ),
    "cache_architecture_native_matrix": (
        "standard_kv",
        "hybrid_ssm_gdn",
        "mixed_swa",
        "cca",
        "minimax_m3_sparse",
        "dsv4_composite",
        "openpangu_native",
    ),
    "turboquant_policy": (
        "q4_default_when_supported",
        "encode_decode_live",
        "explicit_off_honored",
        "unsupported_architecture_exception_honored",
    ),
    "settings_defaults_and_persistence": (
        "bundle_defaults_in_new_ui_session",
        "bundle_defaults_in_api",
        "ui_override_session_scoped",
        "ui_override_restart_persisted",
        "api_request_override_request_scoped",
        "max_context_output_distinct",
        "preview_argv_health_parity",
    ),
    "parser_family_matrix": (
        "reasoning_parsers",
        "tool_parsers",
        "json_xml_boundaries",
        "auto_on_off_policy",
        "no_inline_think_leak",
    ),
    "i18n_katex_responsive_ui": (
        "no_raw_translation_keys",
        "all_supported_locales_checked",
        "minimum_window_width_checked",
        "katex_rendered",
        "currency_preserved",
        "markdown_rendered",
    ),
    "media_image_video_audio": (
        "electron_attachment_flow",
        "raw_api_flow",
        "image",
        "video",
        "audio",
        "media_salt_isolation",
        "post_media_tool_turn",
    ),
    "gateway_lifecycle": (
        "one_model_only_swap",
        "eager_load_on_start",
        "stop_and_disconnect",
        "port_conflict_recovery",
        "lan_rollback",
        "repeated_swap_soak",
    ),
    "family_gemma4_dense_jang": COMMON_FAMILY_ASSERTIONS
    + ("mixed_swa_checked", "image_video_checked", "audio_checked"),
    "family_gemma4_moe_jang": COMMON_FAMILY_ASSERTIONS
    + ("mixed_swa_checked", "image_video_checked", "audio_checked"),
    "family_gemma4_dense_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("mixed_swa_checked", "image_video_checked", "audio_checked"),
    "family_gemma4_moe_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("mixed_swa_checked", "image_video_checked", "audio_checked"),
    "family_qwen_dense_mtp_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("mtp_autodetected", "image_video_checked"),
    "family_qwen_moe_mtp_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("mtp_autodetected", "image_video_checked"),
    "family_qwen_jang": COMMON_FAMILY_ASSERTIONS + ("image_video_checked",),
    "family_qwen_jangtq": COMMON_FAMILY_ASSERTIONS + ("image_video_checked",),
    "family_minimax_m27_jang": COMMON_FAMILY_ASSERTIONS,
    "family_minimax_m27_jangtq": COMMON_FAMILY_ASSERTIONS,
    "family_minimax_m3_jang": COMMON_FAMILY_ASSERTIONS
    + ("native_sparse_cache_checked", "image_video_checked"),
    "family_hy3_jang_mtp": COMMON_FAMILY_ASSERTIONS + ("mtp_autodetected",),
    "family_laguna_s21_jang2l": COMMON_FAMILY_ASSERTIONS
    + ("variable_reasoning_contract_checked", "rotating_swa_cache_checked"),
    "family_laguna_s21_jang4m": COMMON_FAMILY_ASSERTIONS
    + ("variable_reasoning_contract_checked", "rotating_swa_cache_checked"),
    "family_laguna_xs_jang2l": COMMON_FAMILY_ASSERTIONS
    + ("variable_reasoning_contract_checked", "rotating_swa_cache_checked"),
    "family_laguna_xs_jang4m": COMMON_FAMILY_ASSERTIONS
    + ("variable_reasoning_contract_checked", "rotating_swa_cache_checked"),
    "family_laguna_xs_jang6m": COMMON_FAMILY_ASSERTIONS
    + ("variable_reasoning_contract_checked", "rotating_swa_cache_checked"),
    "family_lfm_mxfp": COMMON_FAMILY_ASSERTIONS + ("hybrid_cache_checked",),
    "family_zaya_text_jangtq": COMMON_FAMILY_ASSERTIONS + ("cca_cache_checked",),
    "family_zaya_vl_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("cca_cache_checked", "image_video_checked"),
    "family_bonsai": COMMON_FAMILY_ASSERTIONS
    + ("long_reasoning_checked", "partial_prefix_checked"),
    "family_ornith_jang": COMMON_FAMILY_ASSERTIONS
    + ("long_reasoning_checked", "partial_prefix_checked"),
    "family_ornith_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("long_reasoning_checked", "partial_prefix_checked"),
    "family_dsv4flash_jang": COMMON_FAMILY_ASSERTIONS
    + ("dsv4_composite_cache_checked",),
    "family_dsv4flash_jangtq": COMMON_FAMILY_ASSERTIONS
    + ("dsv4_composite_cache_checked",),
    "family_stepflash": COMMON_FAMILY_ASSERTIONS + ("step_media_checked",),
    "family_nemotron_omni_jangtq": COMMON_FAMILY_ASSERTIONS
    + ("hybrid_cache_checked", "audio_checked", "image_video_checked"),
    "family_nemotron_omni_mxfp": COMMON_FAMILY_ASSERTIONS
    + ("hybrid_cache_checked", "audio_checked", "image_video_checked"),
    "family_openpangu": COMMON_FAMILY_ASSERTIONS + ("openpangu_long_checked",),
    "jang_runtime_provenance": (
        "exact_clean_source",
        "version_commit_tree_exact",
        "bundle_content_hash_parity",
        "laguna_mixed_affine_shape_checked",
    ),
    "release_scope_regression_review": (
        "v1_6_17_to_head_diff_reviewed",
        "all_intended_fixes_mapped",
        "unintended_changes_none_or_documented",
        "public_repository_hygiene_passed",
    ),
}

REQUIRED_CHECKS = tuple(REQUIRED_ASSERTIONS)

REQUIRED_RECORD_KINDS = {
    "exact_source_provenance": ("source_trace",),
    "full_python_suite": ("test_run",),
    "full_panel_suite": ("test_run",),
    "typecheck": ("test_run",),
    "production_build": ("test_run",),
    "electron_visual_multiturn": ("electron_turn",),
    "raw_api_chat": ("api_stream",),
    "raw_api_responses": ("api_stream",),
    "raw_api_anthropic": ("api_stream",),
    "raw_api_ollama": ("api_stream",),
    "interleaved_reasoning_tools": ("electron_turn", "api_stream"),
    "output_integrity_terminal_stats": (
        "electron_turn",
        "api_stream",
    ),
    "cache_paged_off_ssd_partial": ("cache_observation",),
    "cache_paged_on_eviction_refault": ("cache_observation",),
    "cache_restart_and_size_eviction": ("cache_observation",),
    "cache_architecture_native_matrix": ("cache_observation",),
    "turboquant_policy": ("cache_observation",),
    "settings_defaults_and_persistence": (
        "electron_turn",
        "api_stream",
    ),
    "parser_family_matrix": (
        "electron_turn",
        "api_stream",
    ),
    "i18n_katex_responsive_ui": ("electron_turn",),
    "media_image_video_audio": (
        "electron_turn",
        "api_stream",
    ),
    "gateway_lifecycle": ("api_stream",),
    "jang_runtime_provenance": ("source_trace", "test_run"),
    "release_scope_regression_review": ("source_trace",),
}
for _family_check in (name for name in REQUIRED_CHECKS if name.startswith("family_")):
    REQUIRED_RECORD_KINDS[_family_check] = (
        "electron_turn",
        "api_stream",
        "cache_observation",
    )

FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "family_gemma4_dense_jang": {
        "identity_any": ("gemma4", "gemma-4"),
        "quantization": "jang_affine",
        "moe": False,
    },
    "family_gemma4_moe_jang": {
        "identity_any": ("gemma4", "gemma-4"),
        "quantization": "jang_affine",
        "moe": True,
    },
    "family_gemma4_dense_mxfp": {
        "identity_any": ("gemma4", "gemma-4"),
        "quantization": "mxfp",
        "moe": False,
    },
    "family_gemma4_moe_mxfp": {
        "identity_any": ("gemma4", "gemma-4"),
        "quantization": "mxfp",
        "moe": True,
    },
    "family_qwen_dense_mtp_mxfp": {
        "identity_any": ("qwen3_5", "qwen3.6", "qwen-3.6"),
        "identity_none": ("qwen3_5_moe", "a3b"),
        "quantization": "mxfp",
        "moe": False,
        "mtp": True,
    },
    "family_qwen_moe_mtp_mxfp": {
        "identity_any": ("qwen3_5_moe", "qwen3.6", "qwen-3.6"),
        "identity_required_any": ("qwen3_5_moe", "a3b"),
        "quantization": "mxfp",
        "moe": True,
        "mtp": True,
    },
    "family_qwen_jang": {
        "identity_any": ("qwen",),
        "quantization": "jang_affine",
    },
    "family_qwen_jangtq": {
        "identity_any": ("qwen",),
        "quantization": "jangtq",
    },
    "family_minimax_m27_jang": {
        "identity_any": ("minimax_m2", "minimax-m2", "m2.7", "m27"),
        "identity_none": ("minimax_m3", "minimax-m3"),
        "quantization": "jang_affine",
    },
    "family_minimax_m27_jangtq": {
        "identity_any": ("minimax_m2", "minimax-m2", "m2.7", "m27"),
        "identity_none": ("minimax_m3", "minimax-m3"),
        "quantization": "jangtq",
    },
    "family_minimax_m3_jang": {
        "identity_any": ("minimax_m3", "minimax-m3"),
        "quantization": "jang_affine",
        "native_cache_any": ("minimax_m3", "sparse", "lightning"),
    },
    "family_hy3_jang_mtp": {
        "identity_any": ("hy_v3", "hy3"),
        "quantization": "jang_affine",
        "mtp": True,
    },
    "family_laguna_s21_jang2l": {
        "identity_all": ("laguna", "s-2.1"),
        "identity_none": ("xs",),
        "profile_any": ("jang_2l", "jang-2l", "jang2l"),
        "quantization": "jang_affine",
    },
    "family_laguna_s21_jang4m": {
        "identity_all": ("laguna", "s-2.1"),
        "identity_none": ("xs",),
        "profile_any": ("jang_4m", "jang-4m", "jang4m"),
        "quantization": "jang_affine",
    },
    "family_laguna_xs_jang2l": {
        "identity_all": ("laguna", "xs"),
        "profile_any": ("jang_2l", "jang-2l", "jang2l"),
        "quantization": "jang_affine",
    },
    "family_laguna_xs_jang4m": {
        "identity_all": ("laguna", "xs"),
        "profile_any": ("jang_4m", "jang-4m", "jang4m"),
        "quantization": "jang_affine",
    },
    "family_laguna_xs_jang6m": {
        "identity_all": ("laguna", "xs"),
        "profile_any": ("jang_6m", "jang-6m", "jang6m"),
        "quantization": "jang_affine",
    },
    "family_lfm_mxfp": {
        "identity_any": ("lfm2", "lfm"),
        "quantization": "mxfp",
    },
    "family_zaya_text_jangtq": {
        "identity_any": ("zaya",),
        "identity_none": ("zaya1_vl", "zaya-vl"),
        "quantization": "jangtq",
    },
    "family_zaya_vl_mxfp": {
        "identity_required_any": ("zaya1_vl", "zaya-vl"),
        "quantization": "mxfp",
    },
    "family_bonsai": {
        "identity_any": ("bonsai",),
        "quantization": "jang_affine",
    },
    "family_ornith_jang": {
        "identity_any": ("ornith",),
        "quantization": "jang_affine",
    },
    "family_ornith_mxfp": {
        "identity_any": ("ornith",),
        "quantization": "mxfp",
    },
    "family_dsv4flash_jang": {
        "identity_any": ("deepseek_v4", "dsv4", "deepseek-v4"),
        "quantization": "jang_affine",
        "native_cache_any": ("deepseek_v4", "dsv4", "composite"),
    },
    "family_dsv4flash_jangtq": {
        "identity_any": ("deepseek_v4", "dsv4", "deepseek-v4"),
        "quantization": "jangtq",
        "native_cache_any": ("deepseek_v4", "dsv4", "composite"),
    },
    "family_stepflash": {
        "identity_any": ("step3p7", "step-3.7", "step3.7"),
        "quantization": "jang_affine",
    },
    "family_nemotron_omni_jangtq": {
        "identity_all": ("nemotron", "omni"),
        "quantization": "jangtq",
    },
    "family_nemotron_omni_mxfp": {
        "identity_all": ("nemotron", "omni"),
        "quantization": "mxfp",
    },
    "family_openpangu": {
        "identity_any": ("openpangu_v2", "openpangu", "open-pangu"),
        "quantization": "jang_affine",
        "native_cache_any": ("openpangu", "composite"),
    },
}

V5_MANIFEST_SCHEMA = "vmlx-r18-owned-release-preflight-v5"
V5_PRODUCER_ENVELOPE_SCHEMA = "vmlx-r18-owned-producer-envelope-v5"
V5_UI_SCHEMA = "vmlx-r18-owned-ui-capture-v5"
V5_API_SCHEMA = "vmlx-r18-owned-api-capture-v5"
V5_CACHE_SCHEMA = "vmlx-r18-owned-cache-capture-v5"
V5_SESSION_BINDING_SCHEMA = "vmlx-r18-owned-session-binding-v5"
V5_UI_READY_SCHEMA = "vmlx-r18-owned-ui-ready-v5"
V5_UI_RELEASE_SCHEMA = "vmlx-r18-owned-ui-release-v5"
V5_CACHE_PHASE_DONE_SCHEMA = "vmlx-r18-owned-cache-phase-done-v5"
V5_RUN_INTENT_SCHEMA = "vmlx-r18-owned-run-intent-v5"
V5_UI_SESSION_ATTESTATION_SCHEMA = (
    "vmlx-r18-owned-ui-session-attestation-v5"
)
V5_L2_EVICTION_OBSERVATION_SCHEMA = (
    "vmlx-cache-l2-size-eviction-observation-v1"
)
V5_L2_RESTART_OBSERVATION_SCHEMA = (
    "vmlx-cache-l2-restart-restore-observation-v1"
)
V5_L2_SIZE_EVICTION_ATTESTATION_SCHEMA = (
    "vmlx-r18-owned-l2-size-eviction-attestation-v5"
)
V5_RUN_INTENT_HARNESSES = {
    "ui": "panel/scripts/live-real-ui-model-proof.mjs",
    "api": "tests/cross_matrix/run_agentic_protocol_matrix.py",
    "cache": "tests/cross_matrix/run_cache_hierarchy_live_gate.py",
    "semantic": "panel/scripts/scoped-release-preflight-18.py",
}
V5_PRODUCER_NAMES = ("ui", "api", "cache")
V5_PRIMARY_REPRESENTATIVE_ID = "primary_tq_supported"
V5_NATIVE_REPRESENTATIVE_ID = "secondary_native_exception"
V5_REPRESENTATIVE_IDS = (
    V5_PRIMARY_REPRESENTATIVE_ID,
    V5_NATIVE_REPRESENTATIVE_ID,
)
V5_L2_SIZE_EVICTION_REQUIREMENTS = {
    "disk_bytes_within_saved_limit": True,
    "older_unused_prefix_eviction_required": True,
    "recent_target_survival_required": True,
    "restart_restore_required": True,
    "counter_only_evidence_allowed": False,
}
V5_CACHE_PHASES = (
    {
        "index": 0,
        "name": "primary_ssd_only_store",
        "representative_id": V5_PRIMARY_REPRESENTATIVE_ID,
        "bundle_role": "primary",
        "cache_policy": "q4",
        "kv_cache_quantization": "q4",
        "tq_policy": "q4-required",
        "session_policy": "primary_stable_session",
        "paged_ram": False,
        "operation": "store",
        "ui_action_profile": "primary-reasoning-render-store",
        "ui_turn_count": 1,
        "api_action_profile": "full-agentic-plus-cache-store",
        "restart_required": False,
    },
    {
        "index": 1,
        "name": "primary_ssd_only_restart_probe",
        "representative_id": V5_PRIMARY_REPRESENTATIVE_ID,
        "bundle_role": "primary",
        "cache_policy": "q4",
        "kv_cache_quantization": "q4",
        "tq_policy": "q4-required",
        "session_policy": "primary_stable_session",
        "paged_ram": False,
        "operation": "probe",
        "ui_action_profile": "primary-tool-restart-probe",
        "ui_turn_count": 1,
        "api_action_profile": "cache-probe",
        "restart_required": True,
    },
    {
        "index": 2,
        "name": "primary_paged_on_store",
        "representative_id": V5_PRIMARY_REPRESENTATIVE_ID,
        "bundle_role": "primary",
        "cache_policy": "q4",
        "kv_cache_quantization": "q4",
        "tq_policy": "q4-required",
        "session_policy": "primary_stable_session",
        "paged_ram": True,
        "operation": "store-evict-refault",
        "ui_action_profile": "primary-history-paged-evict-refault",
        "ui_turn_count": 1,
        "api_action_profile": "cache-evict-refault",
        "restart_required": True,
    },
    {
        "index": 3,
        "name": "primary_paged_on_restart_probe",
        "representative_id": V5_PRIMARY_REPRESENTATIVE_ID,
        "bundle_role": "primary",
        "cache_policy": "q4",
        "kv_cache_quantization": "q4",
        "tq_policy": "q4-required",
        "session_policy": "primary_stable_session",
        "paged_ram": True,
        "operation": "probe",
        "ui_action_profile": "primary-restart-followup",
        "ui_turn_count": 1,
        "api_action_profile": "cache-restart-probe",
        "restart_required": True,
    },
    {
        "index": 4,
        "name": "primary_tq_off",
        "representative_id": V5_PRIMARY_REPRESENTATIVE_ID,
        "bundle_role": "primary",
        "cache_policy": "ssd-only",
        "kv_cache_quantization": "none",
        "tq_policy": "explicit-off",
        "session_policy": "primary_stable_session",
        "paged_ram": False,
        "operation": "store-probe",
        "ui_action_profile": "primary-tq-off-probe",
        "ui_turn_count": 1,
        "api_action_profile": "cache-tq-off-store-probe",
        "restart_required": True,
    },
    {
        "index": 5,
        "name": "native_exception",
        "representative_id": V5_NATIVE_REPRESENTATIVE_ID,
        "bundle_role": "native",
        "cache_policy": "native",
        "kv_cache_quantization": "none",
        "tq_policy": "native-suppressed",
        "session_policy": "distinct_native_session",
        "paged_ram": False,
        "operation": "switch-validate",
        "ui_action_profile": "native-three-turn-switch",
        "ui_turn_count": 3,
        "api_action_profile": "full-agentic-native-cache",
        "restart_required": True,
    },
)


def _v5_cache_gate_operation(phase: dict[str, Any]) -> str:
    """Map release-policy operations to the cache gate's store/probe CLI."""

    operation = phase.get("operation")
    if operation == "probe":
        return "probe"
    if operation in {
        "store",
        "store-evict-refault",
        "store-probe",
        "switch-validate",
    }:
        return "store"
    raise ValueError(f"unknown v5 cache operation: {operation!r}")


def _v5_cache_gate_scenario(phase: dict[str, Any]) -> str:
    """Select the cache gate scenario required by the release phase.

    The phase-2/phase-3 pair owns the strict L2 size-eviction and restart
    observations.  Leaving the cache harness on its ``standard`` default makes
    those observations structurally impossible, so bind the scenarios to the
    canonical phase indexes rather than relying on a caller-supplied value.
    """

    phase_index = phase.get("index")
    operation = phase.get("operation")
    if phase_index == 2 and operation == "store-evict-refault":
        return "store-evict-refault"
    if phase_index == 3 and operation == "probe":
        return "restart-restore"
    return "standard"


def _v5_derive_l2_size_eviction_attestation(
    *,
    run_id: str,
    nonce: str,
    phase2_summary: dict[str, Any],
    phase2_summary_sha256: str,
    phase3_summary: dict[str, Any],
    phase3_summary_sha256: str,
) -> dict[str, Any]:
    """Derive the strict L2 LRU/restart attestation from raw cache summaries."""

    eviction = phase2_summary.get("l2_size_eviction_observation")
    restart = phase3_summary.get("l2_restart_restore_observation")
    if (
        not isinstance(eviction, dict)
        or eviction.get("schema") != V5_L2_EVICTION_OBSERVATION_SCHEMA
        or not isinstance(restart, dict)
        or restart.get("schema") != V5_L2_RESTART_OBSERVATION_SCHEMA
    ):
        raise ValueError("cache summaries lack strict L2 observations")

    def positive_integer(value: Any, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a positive integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive integer") from exc
        if result <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return result

    def nonnegative_integer(value: Any, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a nonnegative integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a nonnegative integer") from exc
        if result < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
        return result

    def sha256(value: Any, field: str) -> str:
        result = str(value or "")
        if not re.fullmatch(r"[0-9a-f]{64}", result):
            raise ValueError(f"{field} must be a sha256")
        return result

    saved_max_bytes = positive_integer(
        eviction.get("saved_max_bytes"),
        "saved_max_bytes",
    )
    peak_observed_bytes = nonnegative_integer(
        eviction.get("peak_observed_bytes"),
        "peak_observed_bytes",
    )
    final_observed_bytes = nonnegative_integer(
        eviction.get("final_observed_bytes"),
        "final_observed_bytes",
    )
    bounded_filler_request_count = positive_integer(
        eviction.get("bounded_filler_request_count"),
        "bounded_filler_request_count",
    )
    if (
        peak_observed_bytes > saved_max_bytes
        or final_observed_bytes > saved_max_bytes
        or final_observed_bytes > peak_observed_bytes
        or bounded_filler_request_count > 256
    ):
        raise ValueError("L2 size/eviction observation violates its bound")
    old_prefix = sha256(
        eviction.get("old_prefix_fingerprint_sha256"),
        "old_prefix_fingerprint_sha256",
    )
    recent_prefix = sha256(
        eviction.get("recent_prefix_fingerprint_sha256"),
        "recent_prefix_fingerprint_sha256",
    )
    restart_prefix = sha256(
        restart.get("restart_probe_prefix_fingerprint_sha256"),
        "restart_probe_prefix_fingerprint_sha256",
    )
    if old_prefix == recent_prefix or restart_prefix != recent_prefix:
        raise ValueError("L2 prefix identities do not prove LRU survival")
    if (
        eviction.get("old_prefix_evicted") is not True
        or eviction.get("recent_prefix_present") is not True
        or eviction.get("recent_prefix_last_access_after_old") is not True
    ):
        raise ValueError("L2 LRU identity observation is incomplete")
    restart_restored_tokens = positive_integer(
        restart.get("restart_restored_tokens"),
        "restart_restored_tokens",
    )
    restart_disk_blocks = positive_integer(
        restart.get("restart_disk_blocks"),
        "restart_disk_blocks",
    )
    restart_uncached_tokens = nonnegative_integer(
        restart.get("restart_uncached_tokens"),
        "restart_uncached_tokens",
    )
    if restart.get("restart_restore_source") != "block-disk":
        raise ValueError("restart restore is not bound to block-disk")
    for value, field in (
        (phase2_summary_sha256, "phase2_cache_summary_sha256"),
        (phase3_summary_sha256, "phase3_cache_summary_sha256"),
    ):
        sha256(value, field)
    return {
        "schema": V5_L2_SIZE_EVICTION_ATTESTATION_SCHEMA,
        "run_id": run_id,
        "nonce": nonce,
        "store_phase_index": 2,
        "restart_phase_index": 3,
        "phase2_cache_summary_sha256": phase2_summary_sha256,
        "phase3_cache_summary_sha256": phase3_summary_sha256,
        "saved_max_bytes": saved_max_bytes,
        "peak_observed_bytes": peak_observed_bytes,
        "final_observed_bytes": final_observed_bytes,
        "bounded_filler_request_count": bounded_filler_request_count,
        "old_prefix_fingerprint_sha256": old_prefix,
        "recent_prefix_fingerprint_sha256": recent_prefix,
        "old_prefix_evicted": True,
        "recent_prefix_present": True,
        "recent_prefix_last_access_after_old": True,
        "restart_probe_prefix_fingerprint_sha256": restart_prefix,
        "restart_restored_tokens": restart_restored_tokens,
        "restart_disk_blocks": restart_disk_blocks,
        "restart_uncached_tokens": restart_uncached_tokens,
        "restart_restore_source": "block-disk",
    }

# A checkpoint release must close the cross-cutting product gates below.  The
# much larger per-family and seven-architecture campaign remains important, but
# it cannot truthfully be certified by one checkpoint run and therefore is
# surfaced separately as source-owned follow-up scope rather than as rows an
# operator can downgrade or waive in an attestation.
V5_REQUIRED_CHECKS = (
    "exact_source_provenance",
    "full_python_suite",
    "full_panel_suite",
    "typecheck",
    "production_build",
    "electron_visual_multiturn",
    "raw_api_chat",
    "raw_api_responses",
    "raw_api_anthropic",
    "raw_api_ollama",
    "interleaved_reasoning_tools",
    "output_integrity_terminal_stats",
    "cache_paged_off_ssd_partial",
    "cache_paged_on_eviction_refault",
    "cache_restart_and_size_eviction",
    "turboquant_policy",
    "settings_defaults_and_persistence",
    "parser_family_matrix",
    "i18n_katex_responsive_ui",
    "jang_runtime_provenance",
    "release_scope_regression_review",
)
V5_RELEASE_ASSERTIONS = {
    name: REQUIRED_ASSERTIONS[name] for name in V5_REQUIRED_CHECKS
}
V5_FOLLOWUP_CHECKS = tuple(
    name for name in REQUIRED_CHECKS if name not in V5_RELEASE_ASSERTIONS
)

V5_UI_FACT_TO_ASSERTION = {
    "real_start_button": ("electron_visual_multiturn", "real_start_button"),
    "minimum_three_turns": ("electron_visual_multiturn", "minimum_three_turns"),
    "reasoning_rail": ("electron_visual_multiturn", "reasoning_rail_checked"),
    "visible_content": ("electron_visual_multiturn", "visible_content_checked"),
    "tool_result_continuation": (
        "electron_visual_multiturn",
        "tool_result_continuation_checked",
    ),
    "terminal_state": ("electron_visual_multiturn", "terminal_state_checked"),
    "cache_ttft_tps": ("electron_visual_multiturn", "cache_ttft_tps_checked"),
    "rendering": ("electron_visual_multiturn", "rendering_checked"),
    "coherence": ("electron_visual_multiturn", "coherence_checked"),
}


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def run_git(*args: str) -> str:
    return run_git_in(ROOT, *args)


def run_git_in(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_json(
    path: Path,
    failures: list[str],
    *,
    private_label: str | None = None,
) -> dict[str, Any]:
    display = private_label or str(path)
    if not path.exists():
        failures.append(f"missing JSON file: {display}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - preserve exact gate failure
        failures.append(
            f"invalid JSON file: {display}"
            if private_label
            else f"invalid JSON file {display}: {exc}"
        )
        return {}
    if not isinstance(data, dict):
        failures.append(f"JSON file is not an object: {display}")
        return {}
    return data


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def project_version() -> str | None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def locked_project_version() -> str | None:
    text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(
        r'\[\[package\]\]\s+name = "vmlx"\s+version = "([^"]+)"',
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def validate_versions(failures: list[str]) -> dict[str, Any]:
    package = load_json(ROOT / "panel/package.json", failures)
    package_lock = load_json(ROOT / "panel/package-lock.json", failures)
    engine_init = (ROOT / "vmlx_engine/__init__.py").read_text(encoding="utf-8")
    values = {
        "pyproject": project_version(),
        "panel_package": package.get("version"),
        "panel_package_lock": package_lock.get("version"),
        "panel_package_lock_root": nested(
            package_lock,
            "packages",
            "",
            "version",
        ),
        "engine_init": (
            VERSION if f'__version__ = "{VERSION}"' in engine_init else None
        ),
        "uv_lock": locked_project_version(),
    }
    for name, value in values.items():
        require(
            value == VERSION,
            failures,
            f"{name}={value!r}, expected {VERSION}",
        )
    return values


def validate_git_state(failures: list[str]) -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD")
    tree = run_git("rev-parse", "HEAD^{tree}")
    status = run_git("status", "--porcelain", "--untracked-files=all")
    require(not status, failures, "release checkout has tracked or untracked changes")
    release_env_files = [
        ROOT / "panel/.env",
        ROOT / "panel/.env.local",
        ROOT / "panel/.env.production",
        ROOT / "panel/.env.production.local",
    ]
    require(
        not any(path.exists() or path.is_symlink() for path in release_env_files),
        failures,
        "release checkout contains an ignored Electron/Vite environment file",
    )
    require(
        not any(name.startswith("VITE_") for name in os.environ),
        failures,
        "release process contains an untracked VITE_ environment override",
    )
    try:
        upstream = run_git("rev-parse", "@{upstream}")
    except Exception:  # noqa: BLE001 - fail closed without leaking local details
        failures.append("cannot resolve pushed upstream revision")
        upstream = ""
    require(
        upstream == head,
        failures,
        "release HEAD does not match its pushed upstream revision",
    )
    try:
        main_only, branch_only = (
            int(value)
            for value in run_git(
                "rev-list",
                "--left-right",
                "--count",
                "origin/main...HEAD",
            ).split()
        )
    except Exception:  # noqa: BLE001 - fail closed without leaking local details
        failures.append("cannot determine origin/main ancestry")
        main_only, branch_only = -1, -1
    require(
        main_only == 0,
        failures,
        f"release head is missing {main_only} origin/main commit(s)",
    )
    require(
        branch_only == 0,
        failures,
        f"release head has {branch_only} commit(s) not integrated into origin/main",
    )
    try:
        remote_main = run_git("ls-remote", "--exit-code", "origin", "refs/heads/main")
        remote_main_commit = remote_main.split()[0]
    except Exception:  # noqa: BLE001 - fail closed without leaking local details
        failures.append("cannot query live origin/main revision")
        remote_main_commit = ""
    require(
        remote_main_commit == head,
        failures,
        "live origin/main does not match release HEAD",
    )
    try:
        remote_identity = canonical_github_repo(run_git("remote", "get-url", "origin"))
    except Exception:  # noqa: BLE001 - fail closed without leaking local details
        failures.append("cannot resolve vMLX origin repository identity")
        remote_identity = ""
    require(
        remote_identity == "jjang-ai/vmlx",
        failures,
        "vMLX origin is not the canonical public repository",
    )
    return {
        "commit": head,
        "tree": tree,
        "upstream_commit": upstream,
        "remote_main_commit": remote_main_commit,
        "remote_identity": remote_identity,
        "main_only": main_only,
        "branch_only": branch_only,
    }


def path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def canonical_github_repo(url: str) -> str:
    normalized = url.strip()
    scp_match = re.fullmatch(
        r"git@github\.com:([^/:\s]+/[^/:\s]+?)(?:\.git)?",
        normalized,
        re.IGNORECASE,
    )
    if scp_match:
        return scp_match.group(1).lower()
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in {"https", "ssh"}:
        return ""
    if (parsed.hostname or "").lower() != "github.com":
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port is not None or parsed.password is not None:
        return ""
    if scheme == "https" and parsed.username is not None:
        return ""
    if scheme == "ssh" and (parsed.username or "").lower() != "git":
        return ""
    if parsed.query or parsed.fragment:
        return ""
    path = re.sub(r"(?i)\.git$", "", parsed.path.strip("/"))
    if not re.fullmatch(r"[^/:\s]+/[^/:\s]+", path):
        return ""
    return path.lower()


def containing_git_repository(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def validate_private_evidence_root(
    configured: Path | None,
    failures: list[str],
) -> Path | None:
    if configured is None:
        failures.append(
            "missing --private-evidence-root or VMLX_R18_PRIVATE_EVIDENCE_ROOT"
        )
        return None
    lexical = configured.expanduser().absolute()
    resolved = configured.expanduser().resolve()
    public_root = ROOT.resolve()
    outside_public = not path_within(lexical, public_root) and not path_within(
        resolved,
        public_root,
    )
    require(
        outside_public,
        failures,
        "private evidence root must be outside the public repository",
    )
    is_directory = resolved.is_dir()
    require(
        is_directory,
        failures,
        "private evidence root is missing or not a directory",
    )
    outside_any_git = containing_git_repository(resolved) is None
    require(
        outside_any_git,
        failures,
        "private evidence root must not be inside any Git repository",
    )
    return resolved if outside_public and is_directory and outside_any_git else None


def private_artifact_path(
    path: Path,
    private_root: Path,
    failures: list[str],
    label: str,
) -> Path | None:
    lexical = Path(os.path.abspath(path.expanduser()))
    private_root = Path(os.path.abspath(private_root.expanduser()))
    if not path_within(lexical, private_root):
        failures.append(
            f"{label} must remain inside the configured private evidence root"
        )
        return None
    return lexical


def _read_fd_bytes(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file_once(path: Path, root: Path) -> bytes | None:
    """Read one immutable regular file beneath ``root`` without following links.

    Every path component is opened relative to an already-open directory with
    ``O_NOFOLLOW``.  The data is read once from one descriptor, then both that
    descriptor and the directory entry are re-statted.  Callers must hash and
    parse these returned bytes rather than reopening the path.
    """

    lexical_root = Path(os.path.abspath(root.expanduser()))
    lexical_path = Path(os.path.abspath(path.expanduser()))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        return None
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    opened_directories: list[int] = []
    file_fd: int | None = None
    try:
        current_dir_fd = os.open(lexical_root, directory_flags)
        opened_directories.append(current_dir_fd)
        for component in relative.parts[:-1]:
            current_dir_fd = os.open(
                component,
                directory_flags,
                dir_fd=current_dir_fd,
            )
            opened_directories.append(current_dir_fd)

        leaf = relative.parts[-1]
        file_fd = os.open(leaf, file_flags, dir_fd=current_dir_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        raw = _read_fd_bytes(file_fd)
        after = os.fstat(file_fd)
        try:
            entry = os.stat(leaf, dir_fd=current_dir_fd, follow_symlinks=False)
        except OSError:
            return None
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or _stable_file_identity(before) != _stable_file_identity(after)
            or _stable_file_identity(after) != _stable_file_identity(entry)
            or len(raw) != after.st_size
        ):
            return None
        return raw
    except OSError:
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(opened_directories):
            os.close(directory_fd)


def _read_private_evidence_once(
    path: Path,
    private_root: Path,
    failures: list[str],
    label: str,
) -> bytes | None:
    raw = _read_regular_file_once(path, private_root)
    if raw is None:
        failures.append(
            f"{label} is not an immutable single-link regular evidence file"
        )
        return None
    if not raw:
        failures.append(f"{label} is empty")
        return None
    return raw


def _open_absolute_directory_parent(path: Path) -> tuple[int, str] | None:
    """Open every parent component without following links."""

    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_absolute() or len(absolute.parts) < 2:
        return None
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    current_fd: int | None = None
    try:
        current_fd = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:-1]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, absolute.parts[-1]
    except OSError:
        if current_fd is not None:
            os.close(current_fd)
        return None


def _read_bundle_directory_snapshot(bundle_path_value: Any) -> dict[str, Any] | None:
    """Read every config through one locked directory descriptor.

    The directory itself is opened relative to its no-follow parent.  All files
    are then opened relative to that same descriptor, so swapping the pathname
    cannot splice files from a different model into one fingerprint.
    """

    bundle_path = Path(str(bundle_path_value or ""))
    if not bundle_path.is_absolute():
        return None
    parent_and_leaf = _open_absolute_directory_parent(bundle_path)
    if parent_and_leaf is None:
        return None
    parent_fd, leaf = parent_and_leaf
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    bundle_fd: int | None = None
    try:
        bundle_fd = os.open(leaf, directory_flags, dir_fd=parent_fd)
        directory_before = os.fstat(bundle_fd)
        if not stat.S_ISDIR(directory_before.st_mode):
            return None
        files: dict[str, dict[str, Any]] = {}
        raw_files: dict[str, bytes] = {}
        for name in BUNDLE_CONFIG_FILES:
            file_fd: int | None = None
            try:
                file_fd = os.open(name, file_flags, dir_fd=bundle_fd)
                before = os.fstat(file_fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    return None
                raw = _read_fd_bytes(file_fd)
                after = os.fstat(file_fd)
                entry = os.stat(name, dir_fd=bundle_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(entry.st_mode)
                    or entry.st_nlink != 1
                    or _stable_file_identity(before) != _stable_file_identity(after)
                    or _stable_file_identity(after) != _stable_file_identity(entry)
                    or len(raw) != after.st_size
                ):
                    return None
                digest = hashlib.sha256(raw).hexdigest()
                files[name] = {
                    "state": "present",
                    "size_bytes": len(raw),
                    "sha256": digest,
                }
                raw_files[name] = raw
            except OSError:
                return None
            finally:
                if file_fd is not None:
                    os.close(file_fd)

        directory_after = os.fstat(bundle_fd)
        directory_entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        directory_identity = (
            directory_before.st_dev,
            directory_before.st_ino,
            directory_before.st_mode,
            directory_before.st_mtime_ns,
            directory_before.st_ctime_ns,
        )
        if directory_identity != (
            directory_after.st_dev,
            directory_after.st_ino,
            directory_after.st_mode,
            directory_after.st_mtime_ns,
            directory_after.st_ctime_ns,
        ) or (directory_after.st_dev, directory_after.st_ino) != (
            directory_entry.st_dev,
            directory_entry.st_ino,
        ):
            return None

        parsed: dict[str, Any] = {}
        for name in BUNDLE_CONFIG_FILES:
            if name.endswith(".json"):
                try:
                    value = json.loads(raw_files[name].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None
                if not isinstance(value, dict):
                    return None
                parsed[name] = value
            else:
                try:
                    parsed[name] = raw_files[name].decode("utf-8")
                except UnicodeDecodeError:
                    return None
        observed = {
            "schema": "vmlx-bundle-config-v2",
            "model_bundle_path": str(Path(os.path.abspath(bundle_path))),
            "directory_identity": {
                "device": directory_after.st_dev,
                "inode": directory_after.st_ino,
            },
            "files": files,
        }
        # The directory fingerprint is intentionally stronger than the
        # path-free content attestation exposed by /health.  Keep both: the
        # former validates legacy private v2 records, while the latter is the
        # value that can be compared to the running engine without publishing
        # local paths or inode metadata.
        directory_fingerprint = _canonical_json_sha256(observed)
        health_observed = {
            "schema": "vmlx-bundle-config-v1",
            "directory_state": "available",
            "files": files,
        }
        content_fingerprint = _canonical_json_sha256(health_observed)
        return {
            **observed,
            "fingerprint_sha256": content_fingerprint,
            "directory_fingerprint_sha256": directory_fingerprint,
            "health_attestation": {
                **health_observed,
                "aggregate_sha256": content_fingerprint,
                "fingerprint_sha256": content_fingerprint,
            },
            "parsed": parsed,
            "derived": _derive_bundle_facts(parsed),
        }
    finally:
        if bundle_fd is not None:
            os.close(bundle_fd)
        os.close(parent_fd)


def _nested_values(value: Any, key_names: set[str]) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if str(key).lower() in key_names:
                matches.append(nested_value)
            matches.extend(_nested_values(nested_value, key_names))
    elif isinstance(value, list):
        for nested_value in value:
            matches.extend(_nested_values(nested_value, key_names))
    return matches


def _truthy_numeric(value: Any) -> bool:
    return (isinstance(value, bool) and value) or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    )


def _derive_bundle_facts(parsed: dict[str, Any]) -> dict[str, Any]:
    config = parsed.get("config.json")
    jang = parsed.get("jang_config.json")
    generation = parsed.get("generation_config.json")
    tokenizer = parsed.get("tokenizer_config.json")
    if not all(
        isinstance(value, dict) for value in (config, jang, generation, tokenizer)
    ):
        return {}
    identity_values = [
        config.get("model_type"),
        config.get("architectures"),
        config.get("_name_or_path"),
        jang.get("model_family"),
        jang.get("profile"),
        jang.get("quantization"),
    ]
    identity = json.dumps(identity_values, sort_keys=True).lower()
    combined = json.dumps(
        {"config": config, "jang": jang},
        sort_keys=True,
    ).lower()
    if any(token in combined for token in ('"jangtq"', '"mxtq"', "turboquant_codebook")):
        quantization_kind = "jangtq"
    elif any(
        token in combined
        for token in ("mxfp", "mxfp4", "mxfp8", "nvfp4", "mxfp_4", "mxfp_8")
    ):
        quantization_kind = "mxfp"
    elif jang or "affine" in combined or '"jang"' in combined:
        quantization_kind = "jang_affine"
    else:
        quantization_kind = "base_mlx"

    mtp_values = _nested_values(
        {"config": config, "jang": jang},
        {
            "mtp",
            "mtp_depth",
            "num_nextn_predict_layers",
            "num_nextn_predict_tokens",
            "multi_token_prediction",
        },
    )
    mtp = any(
        _truthy_numeric(value)
        or (isinstance(value, str) and value.lower() not in {"", "false", "off", "0"})
        or (isinstance(value, dict) and bool(value))
        for value in mtp_values
    )
    expert_values = _nested_values(
        config,
        {
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "moe_num_experts",
        },
    )
    moe = any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 1
        for value in expert_values
    )
    if any(token in identity for token in ("minimax_m3", "minimax-m3")):
        native_cache = "minimax_m3_sparse"
    elif any(token in identity for token in ("deepseek_v4", "dsv4", "deepseek-v4")):
        native_cache = "dsv4_composite"
    elif any(token in identity for token in ("openpangu", "open-pangu")):
        native_cache = "openpangu_native"
    elif "cca" in combined or "zaya" in identity:
        native_cache = "cca"
    elif any(token in combined for token in ("ssm", "gdn", "hybrid_cache")):
        native_cache = "hybrid_ssm_gdn"
    elif any(token in combined for token in ("sliding_window", "swa")):
        native_cache = "mixed_swa"
    else:
        native_cache = "standard_kv"
    return {
        "identity": identity,
        "quantization_kind": quantization_kind,
        "mtp": mtp,
        "moe": moe,
        "native_cache": native_cache,
        "generation_defaults": {
            key: generation.get(key)
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "max_new_tokens",
                "max_output_tokens",
            )
            if key in generation
        },
    }


def validate_jang_source(failures: list[str]) -> dict[str, Any]:
    configured = os.environ.get("VMLX_JANG_TOOLS_SOURCE")
    require(
        bool(configured),
        failures,
        "VMLX_JANG_TOOLS_SOURCE must identify the exact clean JANG release source",
    )
    if not configured:
        return {}
    root = Path(configured).expanduser().resolve()
    require(
        not path_within(root, ROOT.resolve()),
        failures,
        "JANG release source must be a separate repository",
    )
    require(
        (root / "pyproject.toml").is_file(),
        failures,
        "JANG release source is missing pyproject.toml",
    )
    if not (root / "pyproject.toml").is_file():
        return {}
    try:
        commit = run_git_in(root, "rev-parse", "HEAD")
        tree = run_git_in(root, "rev-parse", "HEAD^{tree}")
        status = run_git_in(root, "status", "--porcelain", "--untracked-files=all")
        upstream = run_git_in(root, "rev-parse", "@{upstream}")
        remote_identity = canonical_github_repo(
            run_git_in(root, "remote", "get-url", "origin")
        )
        remote_main = run_git_in(
            root,
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/heads/main",
        ).split()[0]
    except Exception:  # noqa: BLE001 - fail closed without leaking local details
        failures.append("cannot establish JANG Git provenance")
        return {}
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    version = match.group(1) if match else None
    require(not status, failures, "JANG release source is dirty")
    require(upstream == commit, failures, "JANG source is not synchronized to upstream")
    require(
        remote_identity == "jjang-ai/jangq",
        failures,
        "JANG origin is not the canonical public repository",
    )
    require(
        remote_main == commit,
        failures,
        "live JANG origin/main does not match the pinned release source",
    )
    require(commit == JANG_COMMIT, failures, "JANG source commit is not the .18 pin")
    require(tree == JANG_TREE, failures, "JANG source tree is not the .18 pin")
    require(version == JANG_VERSION, failures, "JANG source version is not the .18 pin")
    return {
        "commit": commit,
        "tree": tree,
        "version": version,
        "upstream_commit": upstream,
        "remote_main_commit": remote_main,
        "remote_identity": remote_identity,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validated_bundle_attestation(
    bundle_path_value: Any,
    health_attestation: Any,
) -> dict[str, str] | None:
    """Compare one directory-fd bundle snapshot with health claims."""

    snapshot = _read_bundle_directory_snapshot(bundle_path_value)
    if snapshot is None:
        return None
    if not isinstance(health_attestation, dict):
        return None
    schema = health_attestation.get("schema")
    if schema == "vmlx-bundle-config-v1":
        expected = snapshot["health_attestation"]
        if (
            health_attestation.get("directory_state") != "available"
            or health_attestation.get("files") != expected["files"]
            or health_attestation.get("aggregate_sha256")
            != expected["aggregate_sha256"]
            or health_attestation.get("fingerprint_sha256")
            != expected["fingerprint_sha256"]
        ):
            return None
    elif schema == "vmlx-bundle-config-v2":
        # Retain compatibility with already-recorded private V5 fixtures while
        # production /health continues to expose only the path-free v1 shape.
        if (
            health_attestation.get("model_bundle_path")
            != snapshot["model_bundle_path"]
            or health_attestation.get("directory_identity")
            != snapshot["directory_identity"]
            or health_attestation.get("files") != snapshot["files"]
            or health_attestation.get("fingerprint_sha256")
            not in {
                snapshot["directory_fingerprint_sha256"],
                snapshot["fingerprint_sha256"],
            }
            or health_attestation.get("derived") != snapshot["derived"]
        ):
            return None
    else:
        return None
    return {
        name: str(snapshot["files"][name]["sha256"])
        for name in BUNDLE_CONFIG_FILES
    }


def _git_head_blob(relative_path: str) -> bytes:
    """Read one blob from HEAD, never from the possibly dirty worktree."""
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"HEAD does not contain required source blob: {relative_path}"
        )
    return completed.stdout


def _git_head_paths(prefix: str, suffix: str | None = None) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", prefix],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot enumerate required HEAD source tree: {prefix}")
    paths = [
        value.strip()
        for value in completed.stdout.splitlines()
        if value.strip() and (suffix is None or value.strip().endswith(suffix))
    ]
    if not paths:
        raise RuntimeError(f"HEAD source tree is empty: {prefix}")
    return sorted(paths)


def _git_head_tree_attestation(
    prefix: str, suffix: str | None = None
) -> dict[str, Any]:
    """Hash path-stable blobs from the release commit."""
    digest = hashlib.sha256()
    count = 0
    for relative_path in _git_head_paths(prefix, suffix):
        content = _git_head_blob(relative_path)
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        count += 1
    return {
        "sha256": digest.hexdigest(),
        "file_count": count,
    }


@lru_cache(maxsize=1)
def release_runtime_source_attestation() -> dict[str, Any]:
    """Return hashes derived exclusively from committed HEAD blobs."""
    python_tree = _git_head_tree_attestation("vmlx_engine", ".py")
    electron_main = _git_head_tree_attestation("panel/src/main")
    renderer = _git_head_tree_attestation("panel/src/renderer")
    commit = run_git("rev-parse", "HEAD")
    tree = run_git("rev-parse", "HEAD^{tree}")
    return {
        "source_commit": commit,
        "source_tree": tree,
        "server_module_sha256": hashlib.sha256(
            _git_head_blob("vmlx_engine/server.py")
        ).hexdigest(),
        "package_init_sha256": hashlib.sha256(
            _git_head_blob("vmlx_engine/__init__.py")
        ).hexdigest(),
        "python_source_tree_sha256": python_tree["sha256"],
        "python_source_file_count": python_tree["file_count"],
        "python_source_read_error_count": 0,
        "electron_main_tree_sha256": electron_main["sha256"],
        "electron_main_file_count": electron_main["file_count"],
        "renderer_source_tree_sha256": renderer["sha256"],
        "renderer_source_file_count": renderer["file_count"],
        "ui_harness_sha256": hashlib.sha256(
            _git_head_blob("panel/scripts/live-real-ui-model-proof.mjs")
        ).hexdigest(),
    }


def _runtime_binding_matches_release_source(binding: Any) -> bool:
    if not isinstance(binding, dict):
        return False
    normalized = _normalized_runtime_binding(binding)
    expected = release_runtime_source_attestation()
    hashes = normalized.get("runtime_source_hashes")
    if not isinstance(hashes, dict):
        return False
    return (
        all(
            hashes.get(field) == expected[field]
            for field in (
                "server_module_sha256",
                "package_init_sha256",
                "python_source_tree_sha256",
            )
        )
        and normalized.get("python_source_file_count")
        == expected["python_source_file_count"]
        and normalized.get("python_source_read_error_count") == 0
        and expected["python_source_read_error_count"] == 0
    )


def _source_observation_matches_release_source(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    expected = release_runtime_source_attestation()
    return all(
        observation.get(field) == expected[field]
        for field in (
            "server_module_sha256",
            "package_init_sha256",
            "python_source_tree_sha256",
            "python_source_file_count",
            "python_source_read_error_count",
        )
    )


def _source_from_semantic_artifact(
    artifact: dict[str, Any],
) -> tuple[str, str, bool]:
    schema = artifact.get("schema")
    if schema in {
        SOURCE_TRACE_SCHEMA,
        COMMAND_RESULT_SCHEMA,
        JANG_RESULT_SCHEMA,
    }:
        source = artifact.get("source")
        if not isinstance(source, dict):
            source = {}
        return (
            str(source.get("commit") or ""),
            str(source.get("tree") or ""),
            source.get("clean") is True,
        )
    if artifact.get("format") == ELECTRON_PROOF_SCHEMA:
        provenance = artifact.get("gitProvenance")
        if not isinstance(provenance, dict):
            provenance = {}
        before = provenance.get("before")
        after = provenance.get("after")
        if not isinstance(before, dict):
            before = {}
        if not isinstance(after, dict):
            after = {}
        stable = (
            before.get("commit") == after.get("commit")
            and before.get("tree") == after.get("tree")
            and before.get("dirty") is False
            and after.get("dirty") is False
        )
        return (
            str(before.get("commit") or ""),
            str(before.get("tree") or ""),
            stable,
        )
    if schema == API_MATRIX_SCHEMA:
        identity = artifact.get("identity")
        if not isinstance(identity, dict):
            identity = {}
        source = identity.get("source")
        if not isinstance(source, dict):
            source = {}
        before = source.get("before")
        after = source.get("after")
        if not isinstance(before, dict):
            before = {}
        if not isinstance(after, dict):
            after = {}
        stable = (
            before.get("head") == after.get("head")
            and before.get("tree") == after.get("tree")
            and before.get("clean") is True
            and after.get("clean") is True
        )
        return (
            str(before.get("head") or ""),
            str(before.get("tree") or ""),
            stable,
        )
    if schema == CACHE_PROOF_SCHEMA:
        identity = artifact.get("identity")
        if not isinstance(identity, dict):
            identity = {}
        source = identity.get("observed_source")
        if not isinstance(source, dict):
            source = {}
        after = artifact.get("observed_source_after")
        if not isinstance(after, dict):
            after = source
        stable = (
            source.get("head") == after.get("head")
            and source.get("tree") == after.get("tree")
            and source.get("dirty") is False
            and after.get("dirty") is False
        )
        return (
            str(source.get("head") or ""),
            str(source.get("tree") or ""),
            stable,
        )
    return "", "", False


def _semantic_source_matches(
    artifact: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    commit, tree, stable = _source_from_semantic_artifact(artifact)
    return (
        stable
        and commit == payload.get("source_commit")
        and tree == payload.get("source_tree")
    )


def _artifact_digest(artifact: dict[str, Any]) -> str:
    return str(artifact.get("__evidence_sha256") or "")


def _artifact_bytes(artifact: dict[str, Any]) -> bytes | None:
    value = artifact.get("__raw_bytes")
    return value if isinstance(value, bytes) else None


def _artifact_for_digest(
    artifacts: list[dict[str, Any]],
    digest: Any,
) -> dict[str, Any] | None:
    normalized = str(digest or "")
    if not SHA256_RE.fullmatch(normalized):
        return None
    matches = [row for row in artifacts if _artifact_digest(row) == normalized]
    return matches[0] if len(matches) == 1 else None


def _semantic_command_result(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Block command claims until this preflight owns the child process.

    Existing command artifacts are author-written receipts.  They cannot prove
    that a PID existed, that it was the process which produced the linked
    bytes, or that the wait status belongs to that process.  This preflight
    does not execute release suites while validating an attestation, so every
    command/full-suite/build claim remains intentionally blocked.
    """

    del artifact, payload, artifacts
    return None


def _semantic_jang_result(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Block JANG claims for the same non-authoritative process-receipt gap."""

    del artifact, payload, artifacts
    return None


def _semantic_test_run(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if artifact.get("schema") == JANG_RESULT_SCHEMA:
        return _semantic_jang_result(artifact, payload, artifacts)
    return _semantic_command_result(artifact, payload, artifacts)


def _semantic_source_trace(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Block source provenance until the producer retains raw live snapshots.

    ``vmlx-r18-source-trace-v1`` is a post-hoc JSON declaration.  It does not
    retain immutable raw bytes for the process start identity, code signature,
    backend argv/import path, listener ownership, health response, CDP
    ``/json`` targets/WebSocket DOM capture, renderer URL, or build manifest.
    File paths and arbitrary PIDs in that declaration are therefore not live
    provenance—even when matching files happen to exist at validation time.
    """

    del artifact, payload, artifacts
    return None


def _semantic_electron_turn(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Fail closed until the UI harness retains process-bound raw CDP evidence.

    ``vmlx-electron-ui-proof-v2`` currently stores a post-hoc summary containing
    persisted messages and coalesced event rows. It does not retain the raw CDP
    target/process observation, the renderer asset loaded by that target, or an
    independent DOM/event stream from which exact visible and reasoning deltas
    can be reconstructed. A parallel ad-hoc schema would only relocate
    self-certification, so no Electron assertion is derived from that summary.
    """

    del artifact, payload, artifacts
    return None


def _tool_call_name(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or call.get("name") or "")
    return str(call.get("name") or "")


def _tool_call_arguments(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    value = (
        function.get("arguments")
        if isinstance(function, dict)
        else call.get("arguments")
    )
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _loopback_base_url(value: Any) -> tuple[str, str, int] | None:
    parsed = urlsplit(str(value or ""))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/v1"}
    ):
        return None
    return parsed.scheme, str(parsed.hostname), int(parsed.port)


def _append_tool_delta(
    calls: dict[int, dict[str, Any]],
    index: int,
    call_id: Any,
    name: Any,
    arguments: Any,
) -> None:
    row = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if call_id:
        row["id"] = str(call_id)
    if name:
        row["name"] += str(name)
    if arguments:
        if isinstance(arguments, str):
            row["arguments"] += arguments
        elif isinstance(arguments, dict):
            row["arguments"] = json.dumps(arguments, separators=(",", ":"))


def _parse_raw_protocol_stream(
    protocol: str,
    raw_bytes: bytes,
) -> dict[str, Any] | None:
    """Parse retained raw SSE/NDJSON bytes without trusting matrix summaries."""

    if protocol not in {"chat", "responses", "anthropic", "ollama"}:
        return None
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not raw_text.strip():
        return None

    parsed_events: list[tuple[str, Any]] = []
    event_name = ""
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            event_name = ""
            continue
        if protocol != "ollama" and line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue
        if protocol != "ollama":
            if not line.startswith("data:"):
                return None
            payload_text = line.removeprefix("data:").strip()
            if payload_text == "[DONE]":
                parsed_events.append((event_name, "[DONE]"))
                continue
        else:
            payload_text = line
        try:
            decoded = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, dict):
            return None
        parsed_events.append((event_name, decoded))
    if not parsed_events:
        return None

    reasoning: list[str] = []
    content: list[str] = []
    tool_parts: dict[int, dict[str, Any]] = {}
    terminals = 0
    terminal_last = False
    normalized_channels: list[str] = []
    event_kinds: list[str] = []
    for event_index, (event_name, data) in enumerate(parsed_events):
        if isinstance(data, dict):
            declared_kind = str(data.get("type") or event_name)
            if (
                declared_kind
                in {
                    "response.failed",
                    "response.cancelled",
                    "response.incomplete",
                    "error",
                }
                or "error" in data
            ):
                return None
        if protocol == "chat":
            if data == "[DONE]":
                terminals += 1
                terminal_last = event_index == len(parsed_events) - 1
                event_kinds.append("DONE")
                continue
            if not isinstance(data, dict):
                return None
            event_kinds.append("chat.chunk")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            reason_delta = delta.get("reasoning_content")
            content_delta = delta.get("content")
            if reason_delta:
                reasoning.append(str(reason_delta))
                normalized_channels.append("reasoning")
            if content_delta:
                content.append(str(content_delta))
                normalized_channels.append("content")
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    return None
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                _append_tool_delta(
                    tool_parts,
                    int(call.get("index") or 0),
                    call.get("id"),
                    function.get("name"),
                    function.get("arguments"),
                )
                normalized_channels.append("tool")
        elif protocol == "responses":
            if not isinstance(data, dict):
                return None
            kind = str(data.get("type") or event_name)
            if event_name and kind != event_name:
                return None
            event_name = kind
            event_kinds.append(event_name)
            if event_name in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                reasoning.append(str(data.get("delta") or ""))
                normalized_channels.append("reasoning")
            elif event_name == "response.output_text.delta":
                content.append(str(data.get("delta") or ""))
                normalized_channels.append("content")
            elif event_name in {
                "response.output_item.added",
                "response.output_item.done",
            }:
                item = data.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    _append_tool_delta(
                        tool_parts,
                        int(data.get("output_index") or len(tool_parts)),
                        item.get("call_id") or item.get("id"),
                        item.get("name"),
                        item.get("arguments"),
                    )
                    normalized_channels.append("tool")
            elif event_name in {"response.completed", "response.incomplete", "error"}:
                if event_name != "response.completed":
                    return None
                response = data.get("response")
                if (
                    not isinstance(response, dict)
                    or response.get("status") != "completed"
                ):
                    return None
                terminals += 1
                terminal_last = event_index == len(parsed_events) - 1
        elif protocol == "anthropic":
            if not isinstance(data, dict):
                return None
            kind = str(data.get("type") or event_name)
            if event_name and kind != event_name:
                return None
            event_kinds.append(kind)
            if kind == "content_block_delta":
                delta = data.get("delta")
                if not isinstance(delta, dict):
                    return None
                delta_type = delta.get("type")
                if delta_type == "thinking_delta":
                    reasoning.append(str(delta.get("thinking") or ""))
                    normalized_channels.append("reasoning")
                elif delta_type == "text_delta":
                    content.append(str(delta.get("text") or ""))
                    normalized_channels.append("content")
                elif delta_type == "input_json_delta":
                    _append_tool_delta(
                        tool_parts,
                        int(data.get("index") or 0),
                        None,
                        None,
                        delta.get("partial_json"),
                    )
                    normalized_channels.append("tool")
            elif kind == "content_block_start":
                block = data.get("content_block")
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    _append_tool_delta(
                        tool_parts,
                        int(data.get("index") or 0),
                        block.get("id"),
                        block.get("name"),
                        block.get("input") or "",
                    )
                    normalized_channels.append("tool")
            elif kind == "message_stop":
                terminals += 1
                terminal_last = event_index == len(parsed_events) - 1
        elif protocol == "ollama":
            if not isinstance(data, dict):
                return None
            event_kinds.append("ollama.done" if data.get("done") is True else "ollama.delta")
            message = data.get("message")
            if isinstance(message, dict):
                if message.get("thinking"):
                    reasoning.append(str(message["thinking"]))
                    normalized_channels.append("reasoning")
                if message.get("content"):
                    content.append(str(message["content"]))
                    normalized_channels.append("content")
                for index, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function") if isinstance(call, dict) else {}
                    if not isinstance(function, dict):
                        return None
                    _append_tool_delta(
                        tool_parts,
                        index,
                        call.get("id") if isinstance(call, dict) else None,
                        function.get("name"),
                        function.get("arguments"),
                    )
                    normalized_channels.append("tool")
            if data.get("done") is True:
                terminals += 1
                terminal_last = event_index == len(parsed_events) - 1
    calls: list[dict[str, Any]] = []
    for _, row in sorted(tool_parts.items()):
        try:
            arguments = json.loads(str(row.get("arguments") or "{}"))
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        calls.append(
            {
                "id": str(row.get("id") or ""),
                "function": {
                    "name": str(row.get("name") or ""),
                    "arguments": arguments,
                },
            }
        )
    if terminals != 1 or not terminal_last:
        return None
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "reasoning_delta_count": len(reasoning),
        "content_delta_count": len(content),
        "tool_calls": calls,
        "terminals": terminals,
        "terminal_last": terminal_last,
        "channels": normalized_channels,
        "event_kinds": event_kinds,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _parse_raw_protocol_stream_v5(
    protocol: str,
    raw_bytes: bytes,
) -> dict[str, Any] | None:
    """Require both a single terminal and a protocol-specific success reason."""

    parsed = _parse_raw_protocol_stream(protocol, raw_bytes)
    if parsed is None:
        return None
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    objects: list[tuple[str, dict[str, Any]]] = []
    event_name = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            event_name = ""
            continue
        if protocol != "ollama" and line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue
        payload = (
            line
            if protocol == "ollama"
            else line.removeprefix("data:").strip()
            if line.startswith("data:")
            else ""
        )
        if not payload or payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        objects.append((event_name, value))
    finish_reasons: list[str] = []
    if protocol == "chat":
        for _, value in objects:
            for choice in value.get("choices") or []:
                if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                    finish_reasons.append(str(choice["finish_reason"]))
        allowed = {"stop", "tool_calls"}
    elif protocol == "responses":
        completed = [
            value
            for event, value in objects
            if str(value.get("type") or event) == "response.completed"
        ]
        if len(completed) != 1:
            return None
        response = completed[0].get("response")
        if not isinstance(response, dict) or response.get("status") != "completed":
            return None
        finish_reasons = ["completed"]
        allowed = {"completed"}
    elif protocol == "anthropic":
        for event, value in objects:
            if str(value.get("type") or event) != "message_delta":
                continue
            delta = value.get("delta")
            if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                finish_reasons.append(str(delta["stop_reason"]))
        allowed = {"end_turn", "tool_use", "stop_sequence"}
    else:
        terminal = [value for _, value in objects if value.get("done") is True]
        if len(terminal) != 1:
            return None
        reason = terminal[0].get("done_reason")
        if reason is not None:
            finish_reasons.append(str(reason))
        allowed = {"stop", "tool_calls"}
    if len(finish_reasons) != 1 or finish_reasons[0] not in allowed:
        return None
    parsed = dict(parsed)
    parsed["finish_reason"] = finish_reasons[0]
    return parsed


def _parse_raw_protocol_nonstream_v5(
    protocol: str,
    raw_bytes: bytes,
) -> dict[str, Any] | None:
    try:
        value = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if protocol == "chat":
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return None
        choice = choices[0]
        if (
            not isinstance(choice, dict)
            or choice.get("finish_reason") not in {"stop", "tool_calls"}
            or not isinstance(choice.get("message"), dict)
        ):
            return None
    elif protocol == "responses":
        if value.get("status") != "completed":
            return None
    elif protocol == "anthropic":
        if value.get("stop_reason") not in {
            "end_turn",
            "tool_use",
            "stop_sequence",
        }:
            return None
    elif protocol == "ollama":
        if value.get("done") is not True:
            return None
    else:
        return None
    if CONTROL_MARKER_RE.search(json.dumps(value, sort_keys=True)):
        return None
    return value


def _request_tool_results(body: Any) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            role = str(value.get("role") or "")
            item_type = str(value.get("type") or "")
            if role == "tool" or item_type in {
                "function_call_output",
                "tool_result",
            }:
                content = value.get("content")
                if content is None:
                    content = value.get("output")
                results.append(
                    {
                        "call_id": str(
                            value.get("tool_call_id")
                            or value.get("call_id")
                            or value.get("tool_use_id")
                            or ""
                        ),
                        "content": str(content or ""),
                    }
                )
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(body)
    return results


def _expected_visible_final(body: Any) -> str:
    strings: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(body)
    patterns = (
        r"(?i)visible answer (?:must be|exactly)\s+[`'\"]?([A-Z0-9][A-Z0-9_.:=/-]{4,})",
        r"(?i)reply exactly\s+[`'\"]?([A-Z0-9][A-Z0-9_.:=/-]{4,})",
    )
    for value in reversed(strings):
        for pattern in patterns:
            match = re.search(pattern, value)
            if match:
                return match.group(1)
    return ""


def _api_flow_facts_from_raw(
    protocol: str,
    request_bodies: list[dict[str, Any]],
    response_bytes: list[bytes],
) -> set[str]:
    facts: set[str] = set()
    if len(request_bodies) != 3 or len(response_bytes) != 3:
        return facts
    parsed = [_parse_raw_protocol_stream(protocol, row) for row in response_bytes]
    if any(row is None for row in parsed):
        return facts
    rounds = [row for row in parsed if isinstance(row, dict)]
    if all(row["terminals"] == 1 and row["terminal_last"] for row in rounds):
        facts.add("terminal_truthful")
    contents = [str(row["content"]) for row in rounds]
    reasoning = [str(row["reasoning"]) for row in rounds]
    if not any(CONTROL_MARKER_RE.search(value) for value in contents):
        facts.add("no_control_markup")
    nonempty_reasoning = [value for value in reasoning if value]
    if (
        len(nonempty_reasoning) >= 2
        and len(
            {hashlib.sha256(value.encode()).hexdigest() for value in nonempty_reasoning}
        )
        == len(nonempty_reasoning)
        and all(
            not (row["reasoning"] and row["reasoning"] == row["content"])
            for row in rounds
        )
    ):
        facts.update({"reasoning_separate", "reasoning_not_stale"})
    expected_final = _expected_visible_final(request_bodies[2])
    if expected_final and contents[2].strip() == expected_final:
        facts.add("nonempty_final")
    if rounds[2]["content_delta_count"] > 1:
        facts.add("content_progressive")
    result_round_2 = _request_tool_results(request_bodies[1])
    result_round_3 = _request_tool_results(request_bodies[2])
    calls = [row["tool_calls"] for row in rounds]
    exact_calls = (
        len(calls[0]) == 1
        and len(calls[1]) == 1
        and not calls[2]
        and _tool_call_name(calls[0][0]) == "file_info"
        and _tool_call_arguments(calls[0][0]) == {"path": "panel/package.json"}
        and _tool_call_name(calls[1][0]) == "run_command"
        and _tool_call_arguments(calls[1][0]) == {"command": "pwd"}
    )
    result_chain = (
        len(result_round_2) >= 1
        and len(result_round_3) >= 2
        and result_round_2[-1]["call_id"] == calls[0][0].get("id")
        and result_round_3[-1]["call_id"] == calls[1][0].get("id")
        and bool(result_round_2[-1]["content"])
        and bool(result_round_3[-1]["content"])
    )
    if result_chain:
        facts.add("history_three_turn")
    if all(
        row.get("stream") is True
        and int(
            row.get("max_output_tokens")
            or row.get("max_tokens")
            or row.get("options", {}).get("num_predict")
            or 0
        )
        > 0
        for row in request_bodies
    ):
        facts.add("request_kwargs")
    if exact_calls and result_chain:
        facts.add("exact_tool_arguments")
        if "nonempty_final" in facts:
            facts.add("tool_result_continuation")
        if (
            reasoning[0]
            and reasoning[1]
            and "reasoning_separate" in facts
            and "tool_result_continuation" in facts
        ):
            facts.add("reasoning_tool_reasoning_tool_answer")
    return facts


def _semantic_api_stream(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Validate the raw protocol capture that the current matrix really emits.

    The v2 harness retains exact decompressed SSE/NDJSON response bytes and a
    metadata manifest.  Those bytes are parsed here directly.  The current
    schema does *not* retain the raw request bodies, raw health responses, or a
    listener-process capture, however.  Therefore it cannot yet bind history,
    kwargs, endpoint ownership, or the loaded bundle path and this semantic
    validator deliberately remains fail-closed after validating what is
    actually available.
    """

    if (
        artifact.get("schema") != API_MATRIX_SCHEMA
        or artifact.get("schema_version") != 2
        or artifact.get("pass") is not True
        or not _semantic_source_matches(artifact, payload)
    ):
        return None
    bases = artifact.get("bases")
    protocols = artifact.get("protocols")
    raw_capture = artifact.get("raw_capture")
    if (
        not isinstance(bases, dict)
        or set(bases) != {"direct", "gateway"}
        or _loopback_base_url(bases.get("direct")) is None
        or _loopback_base_url(bases.get("gateway")) is None
        or _loopback_base_url(bases.get("direct"))
        == _loopback_base_url(bases.get("gateway"))
        or not isinstance(protocols, list)
        or set(protocols) != {"chat", "responses", "anthropic", "ollama"}
        or not isinstance(raw_capture, dict)
        or raw_capture.get("enabled") is not True
        or raw_capture.get("complete") is not True
        or raw_capture.get("capture_layer") != API_CAPTURE_LAYER
        or raw_capture.get("capture_semantics") != API_CAPTURE_SEMANTICS
        or int(raw_capture.get("errors") or 0) != 0
    ):
        return None

    linked = artifacts or []
    manifest = _artifact_for_digest(linked, raw_capture.get("manifest_sha256"))
    if not isinstance(manifest, dict):
        return None
    manifest_core = dict(raw_capture)
    manifest_core.pop("manifest_file", None)
    manifest_core.pop("manifest_sha256", None)
    manifest_public = {
        key: value for key, value in manifest.items() if not str(key).startswith("__")
    }
    if manifest_public != manifest_core:
        return None
    routes = manifest.get("routes")
    if not isinstance(routes, list):
        return None

    expected_labels = {
        f"stream-flow-round{round_number}" for round_number in range(1, 4)
    }
    seen: set[tuple[str, str, str]] = set()
    for route in routes:
        if not isinstance(route, dict):
            return None
        key = (
            str(route.get("base_label") or ""),
            str(route.get("protocol") or ""),
            str(route.get("capture_label") or ""),
        )
        if key[0] not in bases or key[1] not in protocols:
            return None
        if key in seen:
            return None
        seen.add(key)
        if key[2] not in expected_labels:
            continue
        rows = route.get("artifacts")
        if (
            route.get("expected") != 1
            or route.get("started") != 1
            or route.get("finished") != 1
            or route.get("errors") not in ([], None)
            or not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or rows[0].get("verified") is not True
        ):
            return None
        row = rows[0]
        body = _artifact_for_digest(linked, row.get("body_sha256"))
        metadata = _artifact_for_digest(linked, row.get("metadata_sha256"))
        body_bytes = _artifact_bytes(body or {})
        if (
            body_bytes is None
            or hashlib.sha256(body_bytes).hexdigest() != row.get("body_sha256")
            or not isinstance(metadata, dict)
            or metadata.get("capture_layer") != API_CAPTURE_LAYER
            or metadata.get("capture_semantics") != API_CAPTURE_SEMANTICS
            or metadata.get("base_label") != key[0]
            or metadata.get("protocol") != key[1]
            or metadata.get("capture_label") != key[2]
        ):
            return None
        request = metadata.get("request")
        response = metadata.get("response")
        if (
            not isinstance(request, dict)
            or not isinstance(response, dict)
            or request.get("method") != "POST"
            or response.get("status_code") != 200
            or response.get("body_sha256") != row.get("body_sha256")
            or response.get("body_bytes") != len(body_bytes)
            or _parse_raw_protocol_stream(key[1], body_bytes) is None
        ):
            return None

        # The recorder currently stores only the request-body digest.  A raw
        # body with that digest is required before history/kwargs/tool-result
        # continuation can be reconstructed instead of accepted from summary
        # fields.  Normal v2 artifacts therefore stop here.
        request_bytes_artifact = _artifact_for_digest(
            linked,
            request.get("body_sha256"),
        )
        request_bytes = _artifact_bytes(request_bytes_artifact or {})
        if request_bytes is None:
            return None
        try:
            request_body = json.loads(request_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(request_body, dict):
            return None

    required_routes = {
        (base, protocol, label)
        for base in ("direct", "gateway")
        for protocol in protocols
        for label in expected_labels
    }
    if not required_routes.issubset(seen):
        return None

    # Even a future v2 capture that adds request bodies still lacks retained raw
    # /health responses, an lsof/process listener capture, and the actual bundle
    # path plus config-file hashes.  Do not infer those from embedded summaries.
    return None

def _semantic_cache_observation(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Reject cache summaries that are not bound to their retained raw inputs.

    The current cache-v2 summary contains paths to request bodies, SSE, health
    snapshots and cache observations, but it does not carry content digests for
    those paths.  Its tokenizer contract exposes only token counts/digests,
    rather than the token arrays needed to independently recompute LCP.  It also
    lacks a retained engine-log/cache-event stream and a process-bound bundle
    path.  Consequently source-authored row counters cannot certify prefix
    reuse, eviction, restart refault, architecture state, or TurboQuant here.

    The preflight remains blocked until the authoritative cache harness links,
    by SHA-256, each raw request, tokenizer token-array capture, before/after
    health body, SSE body, engine log/cache event, and store/probe lifecycle.
    Separate family artifacts are then required for standard KV, hybrid
    SSM/GDN, mixed SWA, CCA, M3 sparse, DSV4 composite and OpenPangu native
    state; facts are never unioned across models.
    """

    del artifacts
    if (
        artifact.get("schema") != CACHE_PROOF_SCHEMA
        or not _semantic_source_matches(artifact, payload)
    ):
        return None
    return None

def _derive_semantic_record(
    kind: str,
    artifacts: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    validators = {
        "source_trace": _semantic_source_trace,
        "test_run": _semantic_test_run,
        "electron_turn": _semantic_electron_turn,
        "api_stream": _semantic_api_stream,
        "cache_observation": _semantic_cache_observation,
        "media_observation": _semantic_electron_turn,
        "lifecycle_observation": _semantic_api_stream,
    }
    validator = validators.get(kind)
    if validator is None:
        return None
    matches = [
        result
        for artifact in artifacts
        if (result := validator(artifact, payload, artifacts)) is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _derived_assertions_for_check(
    check_name: str,
    semantics_by_kind: dict[str, list[dict[str, Any]]],
) -> set[str]:
    """Map structured observations to this check's assertion vocabulary.

    The mapping is intentionally explicit and incomplete.  A generic passing
    UI/API/cache artifact cannot certify a specialized release row merely
    because its typed proof repeats that row's booleans.  Unsupported claims
    remain uncovered and therefore fail closed until a dedicated raw schema
    and derivation are added.
    """

    def common_facts(kind: str) -> set[str]:
        records = semantics_by_kind.get(kind) or []
        if not records:
            return set()
        sets = [
            {str(fact) for fact in record.get("facts") or [] if isinstance(fact, str)}
            for record in records
        ]
        return set.intersection(*sets) if sets else set()

    def common_api_facts(protocols: set[str]) -> set[str]:
        records = semantics_by_kind.get("api_stream") or []
        if not records:
            return set()
        observed: list[set[str]] = []
        for record in records:
            facts_by_protocol = record.get("facts_by_protocol")
            if not isinstance(facts_by_protocol, dict):
                return set()
            for protocol in protocols:
                facts = facts_by_protocol.get(protocol)
                if not isinstance(facts, list):
                    return set()
                observed.append({str(fact) for fact in facts if isinstance(fact, str)})
        return set.intersection(*observed) if observed else set()

    if check_name == "exact_source_provenance":
        return common_facts("source_trace") & {
            "checkout_head_exact",
            "origin_branch_pushed",
            "origin_main_exact",
            "electron_revision_exact",
            "renderer_revision_exact",
            "python_import_revision_exact",
            "engine_pid_recorded",
            "bundle_config_hashes_recorded",
        }

    if check_name in {
        "full_python_suite",
        "full_panel_suite",
        "typecheck",
        "production_build",
    }:
        allowed = {
            "full_python_suite": {
                "complete_suite",
                "focused_suite_not_substituted",
                "terminal_summary_passed",
            },
            "full_panel_suite": {
                "complete_suite",
                "focused_suite_not_substituted",
                "terminal_summary_passed",
            },
            "typecheck": {"terminal_summary_passed"},
            "production_build": {
                "exact_checkout_built",
                "terminal_summary_passed",
            },
        }
        records = semantics_by_kind.get("test_run") or []
        if any(record.get("result_kind") != check_name for record in records):
            return set()
        return common_facts("test_run") & allowed[check_name]

    if check_name == "jang_runtime_provenance":
        records = semantics_by_kind.get("test_run") or []
        if (
            len(records) != 1
            or records[0].get("result_kind") != "jang_runtime_provenance"
        ):
            return set()
        return common_facts("test_run") & set(REQUIRED_ASSERTIONS[check_name])

    if check_name == "electron_visual_multiturn":
        facts = common_facts("electron_turn")
        mapping = {
            "real_start_button": "real_start_button",
            "minimum_three_turns": "minimum_three_turns",
            "reasoning_rail": "reasoning_rail_checked",
            "visible_content": "visible_content_checked",
            "tool_result_continuation": "tool_result_continuation_checked",
            "terminal_state": "terminal_state_checked",
            "cache_ttft_tps": "cache_ttft_tps_checked",
            "rendering": "rendering_checked",
            "coherence": "coherence_checked",
        }
        return {assertion for fact, assertion in mapping.items() if fact in facts}

    if check_name.startswith("raw_api_"):
        protocol = check_name.removeprefix("raw_api_")
        facts = common_api_facts({protocol})
        mapping = {
            "reasoning_separate": "reasoning_content_separate",
            "content_progressive": "content_progressive",
            "tool_result_continuation": "tool_result_continuation_checked",
            "history_three_turn": "history_checked",
            "request_kwargs": "request_kwargs_checked",
            "terminal_truthful": "terminal_truthful",
        }
        result = {assertion for fact, assertion in mapping.items() if fact in facts}
        if facts:
            # API semantic validation requires both direct and gateway flows.
            result.update({"direct_stream_checked", "gateway_stream_checked"})
        return result

    if check_name == "interleaved_reasoning_tools":
        api = common_api_facts({"chat", "responses", "anthropic", "ollama"})
        ui = common_facts("electron_turn")
        both = api & ui
        mapping = {
            "reasoning_tool_reasoning_tool_answer": (
                "reasoning_tool_reasoning_tool_answer"
            ),
            "exact_tool_arguments": "exact_tool_arguments",
            "no_control_markup": "no_control_markup_leak",
            "nonempty_final": "nonempty_final_answer",
        }
        return {assertion for fact, assertion in mapping.items() if fact in both}

    if check_name == "output_integrity_terminal_stats":
        api = common_api_facts({"chat", "responses", "anthropic", "ollama"})
        ui = common_facts("electron_turn")
        both = api & ui
        result: set[str] = set()
        if "nonempty_final" in both:
            result.add("no_reasoning_only_finalization")
        if {"nonempty_final", "terminal_truthful"} <= api and {
            "nonempty_final",
            "terminal_state",
        } <= ui:
            result.update(
                {
                    "no_eos_or_truncation_regression",
                    "terminal_event_truthful",
                }
            )
        return result

    if check_name == "cache_paged_off_ssd_partial":
        return common_facts("cache_observation") & {
            "paged_ram_disabled",
            "ssd_l2_enabled",
            "longest_prefix_partial_block_hit",
            "uncached_suffix_prefilled",
            "prefill_skip_measured",
            "cross_chat_reuse",
            "cross_session_reuse",
        }

    if check_name == "cache_paged_on_eviction_refault":
        return common_facts("cache_observation") & {
            "paged_ram_enabled",
            "ram_blocks_filled",
            "oldest_unused_evicted",
            "disk_refault_observed",
            "longest_prefix_partial_block_hit",
            "uncached_suffix_prefilled",
        }

    if check_name == "cache_restart_and_size_eviction":
        return common_facts("cache_observation") & {
            "restart_disk_restore",
            "disk_size_limit_enforced",
            "disk_oldest_unused_evicted",
            "ram_percentage_limit_enforced",
            "ram_oom_warning_checked",
        }

    if check_name == "cache_architecture_native_matrix":
        required = set(REQUIRED_ASSERTIONS[check_name])
        records = semantics_by_kind.get("cache_observation") or []
        if len(records) != len(required):
            return set()
        observed: set[str] = set()
        models: set[str] = set()
        bundles: set[str] = set()
        for record in records:
            architecture = set(record.get("facts") or []) & required
            binding = record.get("binding")
            model_id = str(record.get("model_id") or "")
            bundle = (
                str(binding.get("model_bundle_fingerprint_sha256") or "")
                if isinstance(binding, dict)
                else ""
            )
            if (
                len(architecture) != 1
                or record.get("native_state_derived") is not True
                or not SHA256_RE.fullmatch(
                    str(record.get("bundle_snapshot_sha256") or "")
                )
                or not model_id
                or not SHA256_RE.fullmatch(bundle)
                or model_id in models
                or bundle in bundles
            ):
                return set()
            models.add(model_id)
            bundles.add(bundle)
            observed.update(architecture)
        return observed if observed == required else set()

    if check_name == "turboquant_policy":
        return common_facts("cache_observation") & set(REQUIRED_ASSERTIONS[check_name])

    if check_name == "parser_family_matrix":
        api = common_api_facts({"chat", "responses", "anthropic", "ollama"})
        ui = common_facts("electron_turn")
        result = {"no_inline_think_leak"} if "no_control_markup" in api & ui else set()
        return result

    if check_name in {
        "settings_defaults_and_persistence",
        "i18n_katex_responsive_ui",
        "media_image_video_audio",
        "gateway_lifecycle",
        "release_scope_regression_review",
    }:
        return set()

    if check_name.startswith("family_"):
        ui = common_facts("electron_turn")
        api = common_api_facts({"chat", "responses", "anthropic", "ollama"})
        cache = common_facts("cache_observation")
        result: set[str] = set()
        if {"minimum_three_turns", "visible_content", "terminal_state"} <= ui:
            result.add("electron_minimum_three_turns")
        if {"history_three_turn", "terminal_truthful", "nonempty_final"} <= api:
            result.update(
                {
                    "raw_api_minimum_three_turns",
                    "chat_completions_checked",
                    "responses_checked",
                    "anthropic_checked",
                    "ollama_checked",
                }
            )
        if "reasoning_separate" in api and "reasoning_rail" in ui:
            result.add("reasoning_parser_checked")
        if "tool_result_continuation" in api & ui:
            result.add("tool_loop_checked")
        if {
            "paged_ram_disabled",
            "ssd_l2_enabled",
            "longest_prefix_partial_block_hit",
        } <= cache:
            result.add("paged_off_ssd_partial_checked")
        if {
            "paged_ram_enabled",
            "disk_refault_observed",
        } <= cache:
            result.add("paged_on_l2_refault_checked")
        if result:
            result.add("same_source_ui_api_session")
        result.update(
            cache
            & (set(REQUIRED_ASSERTIONS[check_name]) - set(COMMON_FAMILY_ASSERTIONS))
        )
        return result

    # Any future release check fails closed until it has an explicit semantic
    # mapping above.
    return set()


def _normalized_runtime_binding(binding: dict[str, Any]) -> dict[str, Any]:
    runtime_hashes = binding.get("runtime_source_hashes")
    if not isinstance(runtime_hashes, dict):
        runtime_hashes = {
            key: binding.get(key)
            for key in (
                "server_module_sha256",
                "package_init_sha256",
                "python_source_tree_sha256",
            )
        }
    return {
        "backend_pid": binding.get("backend_pid") or binding.get("pid"),
        "runtime_source_hashes": {
            key: runtime_hashes.get(key)
            for key in (
                "server_module_sha256",
                "package_init_sha256",
                "python_source_tree_sha256",
            )
        },
        "model_bundle_fingerprint_sha256": binding.get(
            "model_bundle_fingerprint_sha256"
        ),
        "cache_topology_fingerprint_sha256": binding.get(
            "cache_topology_fingerprint_sha256"
        ),
        "python_source_file_count": binding.get("python_source_file_count"),
        "python_source_read_error_count": binding.get("python_source_read_error_count"),
    }


def _health_family_contract(health: Any) -> dict[str, Any] | None:
    """Bind family facts to the actual bundle directory and its config bytes."""
    if not isinstance(health, dict):
        return None
    if health.get("model_loaded") is not True or health.get("status") != "healthy":
        return None
    bundle = health.get("model_bundle_provenance")
    quantization = health.get("quantization")
    mtp = health.get("mtp")
    routing = health.get("routing")
    native_cache = health.get("native_cache")
    if not isinstance(bundle, dict) or not isinstance(quantization, dict):
        return None
    bundle_hashes = _validated_bundle_attestation(
        health.get("model_bundle_path"),
        bundle,
    )
    if bundle_hashes is None:
        return None
    fingerprint = str(bundle.get("fingerprint_sha256") or "")
    if not SHA256_RE.fullmatch(fingerprint):
        return None
    return {
        "model_name": str(health.get("model_name") or ""),
        "model_type": str(health.get("model_type") or ""),
        "engine_type": str(health.get("engine_type") or ""),
        "model_bundle_fingerprint_sha256": fingerprint,
        "bundle_config_hashes": bundle_hashes,
        "quantization": _attested_mapping(
            quantization,
            (
                "codec",
                "weight_format",
                "backend",
                "profile",
                "mxtq_bits",
                "mxtq_bits_by_role",
                "affine_bits",
                "affine_bits_by_role",
                "target_bits",
                "config_bits",
                "group_size",
                "model_types",
                "architectures",
                "model_family",
                "sidecar",
            ),
        ),
        "mtp": _attested_mapping(
            mtp,
            (
                "family",
                "status",
                "runtime_available",
                "runtime_active",
                "request_policy",
                "issues",
            ),
        ),
        "routing": _attested_mapping(
            routing,
            (
                "trained_active_experts",
                "n_routed_experts",
                "effective_active_experts",
            ),
        ),
        "native_cache": _attested_mapping(
            native_cache,
            (
                "family",
                "schema",
                "cache_type",
                "cache_subtype",
                "components",
                "generic_turboquant_kv",
            ),
        ),
    }


def _attested_mapping(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in fields if field in value}


def _family_identity_text(contract: dict[str, Any]) -> str:
    quantization = contract.get("quantization")
    mtp = contract.get("mtp")
    native_cache = contract.get("native_cache")
    if not isinstance(quantization, dict):
        quantization = {}
    if not isinstance(mtp, dict):
        mtp = {}
    if not isinstance(native_cache, dict):
        native_cache = {}
    values: list[Any] = [
        contract.get("model_name"),
        contract.get("model_type"),
        quantization.get("model_types"),
        quantization.get("architectures"),
        quantization.get("model_family"),
        mtp.get("family"),
        native_cache.get("family"),
        native_cache.get("cache_subtype"),
    ]
    return json.dumps(values, ensure_ascii=True, sort_keys=True).lower()


def _family_quantization_kind(contract: dict[str, Any]) -> str:
    quantization = contract.get("quantization")
    if not isinstance(quantization, dict):
        return "unknown"
    codec = str(quantization.get("codec") or "").lower()
    weight_format = str(quantization.get("weight_format") or "").lower()
    sidecar = quantization.get("sidecar")
    if not isinstance(sidecar, dict):
        sidecar = {}
    if (
        codec == "turboquant_codebook"
        or weight_format in {"mxtq", "jangtq"}
        or sidecar.get("jangtq_runtime") is True
    ):
        return "jangtq"
    if weight_format in {"mxfp4", "mxfp8", "nvfp4", "fp4", "fp8"}:
        return "mxfp"
    if (
        codec == "affine_quantized_matmul"
        and sidecar.get("jang_config") is True
        and sidecar.get("jangtq_runtime") is not True
    ):
        return "jang_affine"
    return "unknown"


def _family_contract_matches(check_name: str, contract: dict[str, Any]) -> bool:
    expected = FAMILY_CONTRACTS.get(check_name)
    if not isinstance(expected, dict):
        return False
    identity = _family_identity_text(contract)
    profile = json.dumps(
        [
            identity,
            (contract.get("quantization") or {}).get("profile"),
            (contract.get("quantization") or {}).get("weight_format"),
        ],
        ensure_ascii=True,
    ).lower()
    native_cache = json.dumps(
        contract.get("native_cache") or {},
        ensure_ascii=True,
        sort_keys=True,
    ).lower()
    if any(token not in identity for token in expected.get("identity_all", ())):
        return False
    if expected.get("identity_any") and not any(
        token in identity for token in expected["identity_any"]
    ):
        return False
    if expected.get("identity_required_any") and not any(
        token in identity for token in expected["identity_required_any"]
    ):
        return False
    if any(token in identity for token in expected.get("identity_none", ())):
        return False
    if expected.get("profile_any") and not any(
        token in profile for token in expected["profile_any"]
    ):
        return False
    if expected.get("native_cache_any") and not any(
        token in native_cache for token in expected["native_cache_any"]
    ):
        return False
    if _family_quantization_kind(contract) != expected.get("quantization"):
        return False
    routing = contract.get("routing")
    if not isinstance(routing, dict):
        routing = {}
    try:
        experts = int(routing.get("n_routed_experts") or 0)
    except (TypeError, ValueError):
        experts = 0
    identity_is_moe = "moe" in identity or "a3b" in identity
    if expected.get("moe") is True and not (experts > 1 or identity_is_moe):
        return False
    if expected.get("moe") is False and (experts > 1 or identity_is_moe):
        return False
    if expected.get("mtp") is True:
        mtp = contract.get("mtp")
        if not isinstance(mtp, dict):
            return False
        if (
            mtp.get("runtime_active") is not True
            or mtp.get("status") != "native_runtime_active"
            or mtp.get("issues") not in ([], None)
        ):
            return False
    return True


def validate_typed_proof_artifact(
    proof: dict[str, Any],
    check_name: str,
    payload: dict[str, Any],
    raw_artifacts: dict[str, dict[str, Any]],
    failures: list[str],
    label: str,
) -> None:
    require(
        proof.get("schema") == PROOF_SCHEMA,
        failures,
        f"{label} typed proof schema is invalid",
    )
    require(
        proof.get("check") == check_name,
        failures,
        f"{label} typed proof check does not match",
    )
    require(
        proof.get("status") == "pass",
        failures,
        f"{label} typed proof status is not pass",
    )
    require(
        proof.get("source_commit") == payload.get("source_commit"),
        failures,
        f"{label} typed proof source_commit does not match",
    )
    require(
        proof.get("source_tree") == payload.get("source_tree"),
        failures,
        f"{label} typed proof source_tree does not match",
    )
    require(
        proof.get("assertions") == payload.get("assertions"),
        failures,
        f"{label} typed proof assertions do not match",
    )
    records = proof.get("records")
    require(
        isinstance(records, list) and bool(records),
        failures,
        f"{label} typed proof has no records",
    )
    if isinstance(records, list):
        require(
            all(isinstance(record, dict) and bool(record) for record in records),
            failures,
            f"{label} typed proof records are malformed",
        )
    if not isinstance(records, list):
        return
    expected_kinds = set(REQUIRED_RECORD_KINDS[check_name])
    observed_kinds: set[str] = set()
    live_session_ids: set[str] = set()
    live_model_ids: set[str] = set()
    live_bindings: set[str] = set()
    live_family_contracts: set[str] = set()
    semantics_by_kind: dict[str, list[dict[str, Any]]] = {}
    all_protocols_checks = {
        "interleaved_reasoning_tools",
        "output_integrity_terminal_stats",
        "settings_defaults_and_persistence",
        "parser_family_matrix",
        "media_image_video_audio",
        "gateway_lifecycle",
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_label = f"{label} record[{index}]"
        kind = record.get("kind")
        if isinstance(kind, str):
            observed_kinds.add(kind)
        require(
            kind in expected_kinds,
            failures,
            f"{record_label} kind is not required for {check_name}",
        )
        require(
            record.get("status") == "pass",
            failures,
            f"{record_label} status is not pass",
        )
        require(
            record.get("source_commit") == payload.get("source_commit"),
            failures,
            f"{record_label} source_commit does not match",
        )
        require(
            record.get("source_tree") == payload.get("source_tree"),
            failures,
            f"{record_label} source_tree does not match",
        )
        record_hashes = record.get("artifact_sha256")
        if isinstance(record_hashes, str):
            record_hashes = [record_hashes]
        hashes_valid = (
            isinstance(record_hashes, list)
            and bool(record_hashes)
            and all(
                isinstance(digest, str) and digest in raw_artifacts
                for digest in record_hashes
            )
        )
        require(
            hashes_valid,
            failures,
            f"{record_label} does not reference verified raw evidence",
        )
        referenced_artifacts = (
            [raw_artifacts[digest] for digest in record_hashes]
            if hashes_valid and isinstance(record_hashes, list)
            else []
        )
        semantic = (
            _derive_semantic_record(kind, referenced_artifacts, payload)
            if isinstance(kind, str)
            else None
        )
        require(
            semantic is not None,
            failures,
            f"{record_label} has no recognized passing semantic artifact",
        )
        if semantic is None:
            continue
        semantics_by_kind.setdefault(str(kind), []).append(semantic)
        command = semantic.get("command")
        require(
            isinstance(command, str) and bool(command.strip()),
            failures,
            f"{record_label} semantic artifact has no command",
        )
        require(
            record.get("command") == command,
            failures,
            f"{record_label} command does not match semantic artifact",
        )
        require(
            semantic.get("exit_code") == 0 and record.get("exit_code") == 0,
            failures,
            f"{record_label} exit_code is not a derived zero",
        )
        recorded_at = semantic.get("recorded_at")
        try:
            timestamp = datetime.fromisoformat(str(recorded_at).replace("Z", "+00:00"))
            timestamp_valid = timestamp.tzinfo is not None
        except ValueError:
            timestamp_valid = False
        require(
            timestamp_valid,
            failures,
            f"{record_label} semantic recorded_at is not timezone-qualified ISO-8601",
        )
        require(
            record.get("recorded_at") == recorded_at,
            failures,
            f"{record_label} recorded_at does not match semantic artifact",
        )
        if kind in {
            "electron_turn",
            "api_stream",
            "cache_observation",
            "media_observation",
            "lifecycle_observation",
        }:
            session_id = semantic.get("model_session_id")
            model_id = semantic.get("model_id")
            require(
                isinstance(session_id, str) and bool(session_id.strip()),
                failures,
                f"{record_label} semantic artifact has no model_session_id",
            )
            require(
                isinstance(model_id, str) and bool(model_id.strip()),
                failures,
                f"{record_label} semantic artifact has no model_id",
            )
            require(
                record.get("model_session_id") == session_id,
                failures,
                f"{record_label} model_session_id does not match semantic artifact",
            )
            require(
                record.get("model_id") == model_id,
                failures,
                f"{record_label} model_id does not match semantic artifact",
            )
            if isinstance(session_id, str) and session_id.strip():
                live_session_ids.add(session_id)
            if isinstance(model_id, str) and model_id.strip():
                live_model_ids.add(model_id)
            binding = semantic.get("binding")
            require(
                isinstance(binding, dict) and bool(binding),
                failures,
                f"{record_label} semantic artifact has no runtime binding",
            )
            if isinstance(binding, dict) and binding:
                live_bindings.add(
                    json.dumps(
                        _normalized_runtime_binding(binding),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            family_contract = semantic.get("family_contract")
            if isinstance(family_contract, dict) and family_contract:
                live_family_contracts.add(
                    json.dumps(
                        family_contract,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        if kind in {"electron_turn", "api_stream"}:
            turn_count = semantic.get("turn_count")
            require(
                isinstance(turn_count, int) and turn_count >= 3,
                failures,
                f"{record_label} semantic artifact has fewer than three turns",
            )
            require(
                record.get("turn_count") == turn_count,
                failures,
                f"{record_label} turn_count does not match semantic artifact",
            )
        if kind == "api_stream":
            protocols = semantic.get("protocols")
            protocol_set = set(protocols) if isinstance(protocols, list) else set()
            if check_name.startswith("raw_api_"):
                required_protocols = {check_name.removeprefix("raw_api_")}
            elif check_name.startswith("family_") or check_name in all_protocols_checks:
                required_protocols = {"chat", "responses", "anthropic", "ollama"}
            else:
                required_protocols = set()
            require(
                required_protocols.issubset(protocol_set),
                failures,
                f"{record_label} is missing required protocol coverage",
            )
            require(
                set(record.get("protocols") or []) == protocol_set,
                failures,
                f"{record_label} protocols do not match semantic artifact",
            )
    derived_assertions = _derived_assertions_for_check(
        check_name,
        semantics_by_kind,
    )
    proof_assertions = proof.get("assertions")
    if not isinstance(proof_assertions, dict):
        proof_assertions = {}
    claimed_assertions = {
        assertion for assertion, value in proof_assertions.items() if value is True
    }
    unsupported_assertions = sorted(claimed_assertions - derived_assertions)
    require(
        not unsupported_assertions,
        failures,
        (
            f"{label} typed proof assertions are not derived from structured raw "
            f"evidence: {unsupported_assertions}"
        ),
    )
    require(
        expected_kinds == observed_kinds,
        failures,
        f"{label} typed proof record kinds mismatch: {sorted(observed_kinds)}",
    )
    if check_name == "cache_architecture_native_matrix":
        expected_architectures = len(REQUIRED_ASSERTIONS[check_name])
        require(
            len(live_model_ids) == expected_architectures,
            failures,
            f"{label} does not contain one distinct model per cache architecture",
        )
        require(
            len(live_bindings) == expected_architectures,
            failures,
            f"{label} does not contain one distinct bound artifact per architecture",
        )
    else:
        require(
            len(live_session_ids) <= 1,
            failures,
            f"{label} live records do not share one model session",
        )
        require(
            len(live_model_ids) <= 1,
            failures,
            f"{label} live records do not share one model identity",
        )
        require(
            len(live_bindings) <= 1,
            failures,
            f"{label} live records do not share one runtime/bundle/cache binding",
        )
    if check_name.startswith("family_"):
        require(
            len(live_family_contracts) == 1,
            failures,
            f"{label} has no single config-derived family contract",
        )
        if len(live_family_contracts) == 1:
            family_contract = json.loads(next(iter(live_family_contracts)))
            require(
                _family_contract_matches(check_name, family_contract),
                failures,
                f"{label} config-derived family/quantization contract does not match",
            )


def verified_evidence_hashes(
    check_name: str,
    payload: dict[str, Any],
    private_root: Path,
    failures: list[str],
) -> tuple[list[str], list[str]]:
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        failures.append(f"{check_name} has no evidence artifacts")
        return [], []
    normalized: list[str] = []
    proof_hashes: list[str] = []
    proofs: list[tuple[dict[str, Any], str]] = []
    raw_artifacts: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(evidence):
        label = f"{check_name} evidence[{index}]"
        if not isinstance(item, dict):
            failures.append(f"{label} is not an object")
            continue
        kind = item.get("kind")
        require(
            kind in {"proof", "raw"},
            failures,
            f"{label} kind must be proof or raw",
        )
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            failures.append(f"{label} has no path")
            continue
        path = private_artifact_path(
            Path(raw_path),
            private_root,
            failures,
            label,
        )
        if path is None:
            continue
        value = item.get("sha256")
        digest = str(value).strip().lower()
        require(
            bool(SHA256_RE.fullmatch(digest)),
            failures,
            f"{label} has invalid SHA-256 digest",
        )
        if not SHA256_RE.fullmatch(digest):
            continue
        artifact_bytes = _read_private_evidence_once(
            path,
            private_root,
            failures,
            label,
        )
        if artifact_bytes is None:
            continue
        actual = hashlib.sha256(artifact_bytes).hexdigest()
        require(
            actual == digest,
            failures,
            f"{label} SHA-256 does not match the file",
        )
        if actual == digest:
            try:
                decoded = json.loads(artifact_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = None
            if kind == "proof":
                if not isinstance(decoded, dict):
                    failures.append(f"{label} is not valid typed JSON proof")
                    continue
                proofs.append((decoded, label))
                proof_hashes.append(digest)
            elif kind == "raw":
                raw_data = decoded if isinstance(decoded, dict) else {}
                raw_data = dict(raw_data)
                raw_data["__evidence_sha256"] = digest
                raw_data["__raw_bytes"] = artifact_bytes
                raw_artifacts[digest] = raw_data
            normalized.append(digest)
    require(
        len(proofs) == 1,
        failures,
        f"{check_name} must have exactly one typed proof artifact",
    )
    require(
        bool(raw_artifacts),
        failures,
        f"{check_name} has no verified raw evidence artifact",
    )
    for proof, proof_label in proofs:
        validate_typed_proof_artifact(
            proof,
            check_name,
            payload,
            raw_artifacts,
            failures,
            proof_label,
        )
    return normalized, proof_hashes


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_aware_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _artifact_in_run_window(artifact: dict[str, Any], payload: dict[str, Any]) -> bool:
    context = payload.get("__validation_context")
    if not isinstance(context, dict):
        return False
    if artifact.get("run_id") != context.get("run_id"):
        return False
    run_started = _parse_aware_timestamp(context.get("started_at"))
    observed_at = _parse_aware_timestamp(context.get("observed_at"))
    artifact_started = _parse_aware_timestamp(artifact.get("started_at"))
    artifact_ended = _parse_aware_timestamp(artifact.get("ended_at"))
    recorded_at = _parse_aware_timestamp(artifact.get("recorded_at"))
    if None in (
        run_started,
        observed_at,
        artifact_started,
        artifact_ended,
        recorded_at,
    ):
        return False
    assert run_started is not None
    assert observed_at is not None
    assert artifact_started is not None
    assert artifact_ended is not None
    assert recorded_at is not None
    return (
        run_started
        <= artifact_started
        <= artifact_ended
        <= recorded_at
        <= observed_at
    )


def _v5_decode_bytes(value: Any) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):  # type: ignore[name-defined]
        return None


def _v5_encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _v5_pin_regular_file(
    path: Path,
    *,
    executable: bool = False,
    allow_readonly_system_hardlink: bool = False,
) -> dict[str, Any]:
    """Open and hash one path without following links.

    This is an operational stale/replacement guard, not a defense against a
    hostile same-user process.  The caller rechecks the same identity after the
    owned child finishes.
    """

    absolute = path.expanduser()
    if not absolute.is_absolute():
        raise ValueError(f"pinned path is not absolute: {path}")
    before = absolute.lstat()
    resolved = absolute.resolve()
    readonly_system_hardlink = bool(
        allow_readonly_system_hardlink
        and before.st_nlink > 1
        and path_within(resolved, Path("/usr/bin"))
        and os.statvfs(resolved).f_flag & os.ST_RDONLY
    )
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or (before.st_nlink != 1 and not readonly_system_hardlink)
        or (executable and not os.access(absolute, os.X_OK))
    ):
        raise ValueError(f"unsafe pinned file: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(absolute, flags)
    try:
        opened = os.fstat(fd)
        if _stable_file_identity(opened) != _stable_file_identity(before):
            raise ValueError(f"pinned file changed while opening: {absolute}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    return {
        "path": str(resolved),
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "size": opened.st_size,
        "mtime_ns": opened.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "readonly_system_hardlink": readonly_system_hardlink,
    }


def _v5_pin_unchanged(pin: dict[str, Any], *, executable: bool = False) -> bool:
    try:
        return _v5_pin_regular_file(
            Path(str(pin.get("path") or "")),
            executable=executable,
            allow_readonly_system_hardlink=bool(
                pin.get("readonly_system_hardlink")
            ),
        ) == pin
    except (OSError, ValueError):
        return False


def _v5_pin_executable_invocation(
    path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pin an executable plus an allowed venv symlink invocation path."""

    absolute = path.expanduser()
    observed = absolute.lstat()
    if not stat.S_ISLNK(observed.st_mode):
        return _v5_pin_regular_file(absolute, executable=True), None
    allowed = path_within(absolute, ROOT / ".venv/bin") or path_within(
        absolute,
        run_dir,
    )
    if not allowed:
        raise ValueError(f"unsafe executable symlink: {absolute}")
    target = absolute.resolve(strict=True)
    executable_pin = _v5_pin_regular_file(target, executable=True)
    return executable_pin, {
        "path": str(absolute),
        "target": os.readlink(absolute),
        "resolved_path": executable_pin["path"],
        "identity": list(_stable_file_identity(observed)),
    }


def _v5_executable_invocation_unchanged(
    invocation: dict[str, Any] | None,
) -> bool:
    if invocation is None:
        return True
    path = Path(str(invocation.get("path") or ""))
    try:
        observed = path.lstat()
        return bool(
            stat.S_ISLNK(observed.st_mode)
            and list(_stable_file_identity(observed))
            == invocation.get("identity")
            and os.readlink(path) == invocation.get("target")
            and str(path.resolve(strict=True)) == invocation.get("resolved_path")
        )
    except OSError:
        return False


def _v5_minimal_env(run_dir: Path, additions: dict[str, str]) -> dict[str, str]:
    """Return a fixed child environment with interpreter/plugin injection cut."""

    allowed_path = "/usr/bin:/bin:/usr/sbin:/sbin"
    env = {
        "PATH": allowed_path,
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": str(run_dir),
    }
    forbidden_prefixes = (
        "BASH_",
        "DYLD_",
        "GIT_",
        "LD_",
        "NODE",
        "NPM",
        "PERL5",
        "PYTHON",
        "PYTEST",
        "RUBY",
        "npm_",
    )
    forbidden_exact = {
        "BASH_ENV",
        "ENV",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SHELLOPTS",
        "TMPDIR",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "PIP_CONFIG_FILE",
        "PIP_REQUIRE_VIRTUALENV",
        "SITE_CUSTOMIZE",
        "ZDOTDIR",
    }
    for key, value in additions.items():
        if key in forbidden_exact or key.startswith(forbidden_prefixes):
            raise ValueError(f"unsafe child environment key: {key}")
        env[str(key)] = str(value)
    return env


def _v5_make_run_directory(private_root: Path, run_id: str, nonce: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", run_id):
        raise ValueError("run ID contains unsafe characters")
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise ValueError("run nonce is invalid")
    run_dir = private_root / f"{run_id}-{nonce}"
    run_dir.mkdir(mode=0o700)
    opened = run_dir.lstat()
    if not stat.S_ISDIR(opened.st_mode) or stat.S_ISLNK(opened.st_mode):
        raise ValueError("exclusive run output is not a real directory")
    return run_dir


def _v5_open_exclusive_capture(run_dir: Path, name: str) -> tuple[Path, int]:
    if not re.fullmatch(r"[a-z0-9_.-]+", name):
        raise ValueError("unsafe capture name")
    path = run_dir / name
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    return path, fd


def _v5_owned_coordination_path(run_dir: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_.-]+", name):
        raise ValueError("unsafe coordination filename")
    path = run_dir / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"stale coordination path exists: {name}")
    return path


def _v5_phase_coordination_paths(
    control_dir: Path,
    phase: dict[str, Any],
) -> dict[str, Path]:
    index = phase.get("index")
    if not isinstance(index, int) or index < 0:
        raise ValueError("cache phase has no safe index")
    prefix = f"ui.phase-{index:02d}"
    return {
        "binding": _v5_owned_coordination_path(
            control_dir,
            f"{prefix}.session.json",
        ),
        "ready": _v5_owned_coordination_path(
            control_dir,
            f"{prefix}.ready.json",
        ),
        "release": _v5_owned_coordination_path(
            control_dir,
            f"{prefix}.release.json",
        ),
        "cache_done": _v5_owned_coordination_path(
            control_dir,
            f"cache.phase-{index:02d}.done.json",
        ),
        "ui_attestation": _v5_owned_coordination_path(
            control_dir,
            f"{prefix}.attestation.json",
        ),
        "paired_api": _v5_owned_coordination_path(
            control_dir,
            f"api.phase-{index:02d}.matrix.json",
        ),
    }


def _v5_existing_phase_paths(
    control_dir: Path,
    phase: dict[str, Any],
) -> dict[str, Path]:
    """Return phase paths without accepting aliases or traversals."""

    index = phase.get("index")
    if not isinstance(index, int) or index < 0:
        raise ValueError("cache phase has no safe index")
    names = {
        "binding": f"ui.phase-{index:02d}.session.json",
        "ready": f"ui.phase-{index:02d}.ready.json",
        "release": f"ui.phase-{index:02d}.release.json",
        "cache_done": f"cache.phase-{index:02d}.done.json",
        "ui_attestation": f"ui.phase-{index:02d}.attestation.json",
        "paired_api": f"api.phase-{index:02d}.matrix.json",
    }
    result: dict[str, Path] = {}
    root = control_dir.resolve()
    for key, name in names.items():
        path = control_dir / name
        if path.parent.resolve() != root:
            raise ValueError("phase path escaped its control directory")
        result[key] = path
    return result


def _v5_write_exclusive_json(path: Path, value: dict[str, Any]) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or _stable_file_identity(opened) != _stable_file_identity(observed)
    ):
        raise RuntimeError(f"exclusive coordination file was replaced: {path.name}")
    return payload


def _v5_read_owned_json(path: Path) -> tuple[dict[str, Any], bytes]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RuntimeError(f"unsafe coordination file: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _stable_file_identity(opened) != _stable_file_identity(before):
            raise RuntimeError(f"coordination file changed while opening: {path.name}")
        payload = _read_fd_bytes(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()
    if (
        _stable_file_identity(opened) != _stable_file_identity(after)
        or _stable_file_identity(after) != _stable_file_identity(current)
    ):
        raise RuntimeError(f"coordination file changed while reading: {path.name}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"coordination file is not JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"coordination file is not an object: {path.name}")
    return value, payload


def _v5_process_matches_pin(
    process_observation: dict[str, Any] | None,
    executable_pin: dict[str, Any],
) -> bool:
    return bool(
        isinstance(process_observation, dict)
        and process_observation.get("executable_path") == executable_pin["path"]
        and process_observation.get("executable_sha256") == executable_pin["sha256"]
        and process_observation.get("start_identity")
    )


def _v5_start_owned_child(
    name: str,
    spec: dict[str, Any],
    run_context: dict[str, Any],
    run_dir: Path,
    *,
    replacements: dict[str, str] | None = None,
    output_basename: str | None = None,
) -> dict[str, Any]:
    """Start one pinned producer while retaining its parent-owned output FD."""

    argv = [str(value) for value in spec["argv"]]
    if not argv:
        raise ValueError(f"{name} has no argv")
    executable_pin, executable_invocation = _v5_pin_executable_invocation(
        Path(argv[0]),
        run_dir,
    )
    script_pins = [
        _v5_pin_regular_file(Path(value))
        for value in argv[1:]
        if isinstance(value, str)
        and value.startswith("/")
        and Path(value).is_file()
        and not os.access(value, os.X_OK)
    ]
    output_path, output_fd = _v5_open_exclusive_capture(
        run_dir,
        output_basename or f"{name}.producer.json",
    )
    token_values = {
        "{OUTPUT_FD}": str(output_fd),
        "{RUN_ID}": str(run_context["run_id"]),
        "{NONCE}": str(run_context["nonce"]),
        **(replacements or {}),
    }
    child_argv: list[str] = []
    for value in argv:
        expanded = value
        for token, replacement in token_values.items():
            expanded = expanded.replace(token, replacement)
        if re.search(r"\{[A-Z0-9_]+\}", expanded):
            os.close(output_fd)
            output_path.unlink(missing_ok=True)
            raise ValueError(f"{name} child argv has an unresolved placeholder")
        child_argv.append(expanded)
    env = _v5_minimal_env(run_dir, dict(spec.get("env") or {}))
    started_at = _iso_now()
    try:
        process = subprocess.Popen(  # noqa: S603 - static source-owned plan
            child_argv,
            cwd=Path(spec["cwd"]).resolve(),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(output_fd,),
        )
    except BaseException:
        os.close(output_fd)
        output_path.unlink(missing_ok=True)
        raise
    observation = _observe_process(process.pid)
    if not _v5_process_matches_pin(observation, executable_pin):
        process.kill()
        process.wait()
        os.close(output_fd)
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"{name} child executable/start identity mismatch")
    return {
        "name": name,
        "process": process,
        "output_fd": output_fd,
        "output_path": output_path,
        "argv": child_argv,
        "cwd": str(Path(spec["cwd"]).resolve()),
        "started_at": started_at,
        "executable": executable_pin,
        "executable_invocation": executable_invocation,
        "scripts": script_pins,
        "process_observation": observation,
    }


def _v5_abort_owned_child(handle: dict[str, Any]) -> None:
    process = handle.get("process")
    if isinstance(process, subprocess.Popen):
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=5)
    output_fd = handle.pop("output_fd", None)
    if isinstance(output_fd, int):
        with contextlib.suppress(OSError):
            os.close(output_fd)


def _v5_finish_owned_child(
    handle: dict[str, Any],
    run_context: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Finish one producer and validate its exact parent-owned envelope."""

    name = str(handle["name"])
    process = handle["process"]
    output_fd = int(handle["output_fd"])
    output_path = Path(handle["output_path"])
    try:
        stdout, stderr = process.communicate(input=b"finish\n", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _v5_abort_owned_child(handle)
        raise RuntimeError(f"{name} owned producer timed out") from exc
    ended_at = _iso_now()
    try:
        os.fsync(output_fd)
        os.lseek(output_fd, 0, os.SEEK_SET)
        capture = _read_fd_bytes(output_fd)
        capture_stat = os.fstat(output_fd)
    finally:
        os.close(output_fd)
        handle.pop("output_fd", None)
    executable_pin = handle["executable"]
    executable_invocation = handle.get("executable_invocation")
    script_pins = handle["scripts"]
    executable_stable = _v5_pin_unchanged(executable_pin, executable=True)
    invocation_stable = _v5_executable_invocation_unchanged(
        executable_invocation
    )
    scripts_stable = all(_v5_pin_unchanged(pin) for pin in script_pins)
    if (
        process.returncode != 0
        or not capture
        or not executable_stable
        or not invocation_stable
        or not scripts_stable
    ):
        stdout_path = _v5_write_private_child_stream_capture(
            output_path,
            "stdout",
            stdout,
        )
        stderr_path = _v5_write_private_child_stream_capture(
            output_path,
            "stderr",
            stderr,
        )
        raise RuntimeError(
            f"{name} owned producer failed or changed "
            f"(exit={process.returncode}, capture_bytes={len(capture)}, "
            f"executable_stable={executable_stable}, "
            f"invocation_stable={invocation_stable}, "
            f"scripts_stable={scripts_stable}, stderr_sha256="
            f"{hashlib.sha256(stderr).hexdigest()}, "
            f"stdout_path={stdout_path or 'none'}, "
            f"stderr_path={stderr_path or 'none'})"
        )
    path_stat = output_path.lstat()
    if (
        _stable_file_identity(path_stat) != _stable_file_identity(capture_stat)
        or path_stat.st_nlink != 1
        or not stat.S_ISREG(path_stat.st_mode)
    ):
        raise RuntimeError(f"{name} capture path was replaced")
    try:
        envelope = json.loads(capture.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{name} producer output is invalid JSON") from exc
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema") != V5_PRODUCER_ENVELOPE_SCHEMA
        or envelope.get("producer") != name
        or envelope.get("run_id") != run_context["run_id"]
        or envelope.get("nonce") != run_context["nonce"]
    ):
        raise RuntimeError(f"{name} producer envelope binding mismatch")
    return {
        "name": name,
        "pid": process.pid,
        "argv": handle["argv"],
        "cwd": handle["cwd"],
        "started_at": handle["started_at"],
        "ended_at": ended_at,
        "exit_code": process.returncode,
        "executable": executable_pin,
        "executable_invocation": executable_invocation,
        "scripts": script_pins,
        "process_observation": handle["process_observation"],
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "capture_path": str(output_path),
        "capture_sha256": hashlib.sha256(capture).hexdigest(),
        "capture": envelope,
    }


def _v5_write_private_child_stream_capture(
    output_path: Path,
    suffix: str,
    payload: bytes,
) -> str | None:
    """Retain a failed producer stream in the private run directory.

    The release manifest still carries only hashes.  The path is emitted only
    when a producer fails before writing a valid owned envelope, so the operator
    can inspect the root exception without weakening public artifact hygiene.
    """

    if not payload:
        return None
    if suffix not in {"stdout", "stderr"}:
        raise ValueError("unsafe child stream suffix")
    capture_name = f"{output_path.stem}.{suffix}"
    capture_path, fd = _v5_open_exclusive_capture(output_path.parent, capture_name)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return str(capture_path)


def _v5_run_owned_child(
    name: str,
    spec: dict[str, Any],
    run_context: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Launch one allowlisted child and retain its exact output bytes."""

    handle = _v5_start_owned_child(name, spec, run_context, run_dir)
    try:
        return _v5_finish_owned_child(handle, run_context)
    except BaseException:
        _v5_abort_owned_child(handle)
        raise


def _v5_loopback_http_get(url: str, *, timeout: float = 10.0) -> bytes:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"non-loopback observation URL: {url}")
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port,
        timeout=timeout,
    )
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"observation GET {path} returned {response.status}")
        return body
    finally:
        connection.close()


def _v5_websocket_frame(payload: bytes) -> bytes:
    mask = secrets.token_bytes(4)
    length = len(payload)
    if length < 126:
        header = bytes((0x81, 0x80 | length))
    elif length < 65536:
        header = bytes((0x81, 0x80 | 126)) + struct.pack("!H", length)
    else:
        header = bytes((0x81, 0x80 | 127)) + struct.pack("!Q", length)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return header + mask + masked


def _v5_recv_exact(stream: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise RuntimeError("CDP websocket closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _v5_recv_websocket_frame(stream: socket.socket) -> tuple[int, bytes]:
    first, second = _v5_recv_exact(stream, 2)
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _v5_recv_exact(stream, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _v5_recv_exact(stream, 8))[0]
    mask = _v5_recv_exact(stream, 4) if second & 0x80 else b""
    payload = _v5_recv_exact(stream, length)
    if mask:
        payload = bytes(
            value ^ mask[index % 4] for index, value in enumerate(payload)
        )
    return opcode, payload


def _v5_cdp_dom_snapshot(cdp_base_url: str, *, timeout: float = 10.0) -> bytes:
    parsed = urlsplit(cdp_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
    ):
        raise ValueError("CDP base URL must be loopback HTTP")
    targets_raw = _v5_loopback_http_get(
        f"http://{parsed.hostname}:{parsed.port}/json/list",
        timeout=timeout,
    )
    targets = json.loads(targets_raw)
    pages = [
        item
        for item in targets
        if isinstance(item, dict)
        and item.get("type") == "page"
        and isinstance(item.get("webSocketDebuggerUrl"), str)
    ]
    if len(pages) != 1:
        raise RuntimeError("CDP did not expose exactly one page target")
    websocket_url = urlsplit(pages[0]["webSocketDebuggerUrl"])
    if (
        websocket_url.scheme != "ws"
        or websocket_url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or websocket_url.port is None
    ):
        raise RuntimeError("CDP websocket is not loopback")
    stream = socket.create_connection(
        (websocket_url.hostname, websocket_url.port),
        timeout=timeout,
    )
    try:
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {websocket_url.path} HTTP/1.1\r\n"
            f"Host: {websocket_url.hostname}:{websocket_url.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        stream.sendall(request)
        handshake = b""
        while b"\r\n\r\n" not in handshake:
            handshake += stream.recv(4096)
            if len(handshake) > 65536:
                raise RuntimeError("oversized CDP websocket handshake")
        if not handshake.startswith(b"HTTP/1.1 101"):
            raise RuntimeError("CDP websocket upgrade failed")
        def cdp_command(
            command_id: int,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            command = json.dumps(
                {
                    "id": command_id,
                    "method": method,
                    "params": params,
                },
                separators=(",", ":"),
            ).encode()
            stream.sendall(_v5_websocket_frame(command))
            while True:
                opcode, payload = _v5_recv_websocket_frame(stream)
                if opcode == 0x9:
                    stream.sendall(bytes((0x8A, len(payload))) + payload)
                    continue
                if opcode == 0x8:
                    raise RuntimeError("CDP websocket closed before response")
                if opcode != 0x1:
                    continue
                decoded = json.loads(payload)
                if decoded.get("id") != command_id:
                    continue
                if decoded.get("error") or nested(
                    decoded,
                    "result",
                    "exceptionDetails",
                ):
                    raise RuntimeError(f"CDP command failed: {method}")
                return decoded

        metrics_result = cdp_command(
            1,
            "Runtime.evaluate",
            {
                "expression": (
                    "JSON.stringify({width:window.innerWidth,"
                    "height:window.innerHeight,"
                    "deviceScaleFactor:window.devicePixelRatio||1})"
                ),
                "returnByValue": True,
            },
        )
        metrics_value = nested(metrics_result, "result", "result", "value")
        try:
            prior_metrics = json.loads(metrics_value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("CDP did not expose prior viewport metrics") from exc
        if (
            not isinstance(prior_metrics, dict)
            or not isinstance(prior_metrics.get("width"), int)
            or prior_metrics["width"] <= 0
            or not isinstance(prior_metrics.get("height"), int)
            or prior_metrics["height"] <= 0
            or not isinstance(
                prior_metrics.get("deviceScaleFactor"),
                (int, float),
            )
            or prior_metrics["deviceScaleFactor"] <= 0
        ):
            raise RuntimeError("CDP prior viewport metrics are invalid")
        cdp_command(
            2,
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 640,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        expression = (
            "(async()=>{"
            "const wait=(ms)=>new Promise((resolve)=>setTimeout(resolve,ms));"
            "const visible=(node)=>Boolean(node&&node.getClientRects().length);"
            "const catalog=await globalThis.api?.i18n?.getCatalogContract?.();"
            "if(!catalog||!Array.isArray(catalog.supportedLocales)||"
            "!Array.isArray(catalog.translationKeys)||"
            "catalog.supportedLocales.length===0||catalog.translationKeys.length===0)"
            "throw new Error('authoritative locale catalog missing');"
            "const supportedLocales=[...new Set(catalog.supportedLocales.map(String))];"
            "const translationKeys=[...new Set(catalog.translationKeys.map(String))];"
            "const canonicalKeySet=new Set(translationKeys.map((key)=>key.toLowerCase()));"
            "const rawTranslationKeys=(text)=>[...new Set("
            "(String(text).match(/[A-Za-z][A-Za-z0-9_-]*(?:\\.[A-Za-z0-9_-]+)+/g)||[])"
            ".filter((token)=>canonicalKeySet.has(token.toLowerCase())))].sort();"
            "const initial=localStorage.getItem('vmlx-locale')||'en';"
            "const localeEvidence=[];"
            "let visibleLocaleOptions=[];"
            "let pickerSupportedLocales=[];"
            "for(const locale of supportedLocales){"
            "const picker=[...document.querySelectorAll('[data-vmlx-locale-picker]')]"
            ".find((button)=>visible(button));"
            "if(!picker)throw new Error('visible language picker missing');"
            "picker.click();await wait(80);"
            "const options=[...document.querySelectorAll('[data-vmlx-locale-option]')]"
            ".filter((button)=>visible(button));"
            "visibleLocaleOptions=[...new Set(options.map((button)=>"
            "button.getAttribute('data-vmlx-locale-option')||'').filter(Boolean))].sort();"
            "pickerSupportedLocales=[...new Set(String("
            "picker.getAttribute('data-vmlx-supported-locales')||'').split(',')"
            ".filter(Boolean))].sort();"
            "const option=options.find((button)=>"
            "button.getAttribute('data-vmlx-locale-option')===locale);"
            "if(!option)throw new Error('visible locale option missing:'+locale);"
            "option.click();"
            "for(let i=0;i<40&&localStorage.getItem('vmlx-locale')!==locale;i++)"
            "await wait(25);"
            "await wait(80);"
            "const text=document.body?.innerText||'';"
            "localeEvidence.push({locale,"
            "selected_locale:localStorage.getItem('vmlx-locale')||'',"
            "raw_translation_keys:rawTranslationKeys(text).slice(0,64)});"
            "}"
            "if(supportedLocales.includes(initial)){"
            "const picker=[...document.querySelectorAll('[data-vmlx-locale-picker]')]"
            ".find((button)=>visible(button));"
            "picker?.click();await wait(80);"
            "const option=[...document.querySelectorAll('[data-vmlx-locale-option]')]"
            ".find((button)=>visible(button)&&"
            "button.getAttribute('data-vmlx-locale-option')===initial);"
            "option?.click();await wait(100);"
            "}"
            "const sessions=await globalThis.api?.sessions?.list?.()||"
            "await globalThis.window?.api?.sessions?.list?.()||[];"
            "const messageRoots=[...document.querySelectorAll("
            "'[data-vmlx-proof-message-role=\"assistant\"]')].slice(-3);"
            "const messages=messageRoots.map((root)=>{"
            "const answer=root.querySelector('[data-vmlx-proof-answer=\"true\"]');"
            "const reasoning=[...root.querySelectorAll("
            "'[data-vmlx-proof-reasoning-content=\"true\"]')]"
            ".map((node)=>(node.textContent||'').trim()).filter(Boolean).join('\\n');"
            "return {"
            "message_id:root.getAttribute('data-vmlx-proof-message-id')||'',"
            "reasoning_text:reasoning,"
            "content_text:(answer?.textContent||'').trim(),"
            "html:answer?.innerHTML||''"
            "};"
            "});"
            "return JSON.stringify({"
            "url:location.href,"
            "text:document.body?.innerText||'',"
            "html:document.body?.innerHTML||'',"
            "messages,"
            "locales:localeEvidence,"
            "locale_catalog_source:'main_ipc_canonical_locale_json',"
            "supported_locales:supportedLocales,"
            "picker_supported_locales:pickerSupportedLocales,"
            "visible_locale_options:visibleLocaleOptions,"
            "translation_key_count:translationKeys.length,"
            "catalog_translation_keys:translationKeys,"
            "raw_translation_keys:rawTranslationKeys(document.body?.innerText||'').slice(0,64),"
            "viewport:{width:window.innerWidth,"
            "scroll_width:Math.max(document.documentElement?.scrollWidth||0,"
            "document.body?.scrollWidth||0)},"
            "sourceCommit:globalThis.__VMLINUX_SOURCE_COMMIT__||"
            "document.documentElement.dataset.sourceCommit||'',"
            "session_ids:Array.isArray(sessions)?sessions.map((row)=>row?.id)"
            ".filter(Boolean):[]"
            "});"
            "})()"
        )
        value: str | None = None
        try:
            decoded = cdp_command(
                3,
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
            value = nested(decoded, "result", "result", "value")
            if not isinstance(value, str):
                raise RuntimeError("CDP DOM evaluation returned no value")
        finally:
            cdp_command(
                4,
                "Emulation.clearDeviceMetricsOverride",
                {},
            )
        restored_result = cdp_command(
            5,
            "Runtime.evaluate",
            {
                "expression": "JSON.stringify({width:window.innerWidth,height:window.innerHeight})",
                "returnByValue": True,
            },
        )
        restored_value = nested(restored_result, "result", "result", "value")
        try:
            restored_metrics = json.loads(restored_value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("CDP did not expose restored viewport metrics") from exc
        if (
            restored_metrics.get("width") != prior_metrics["width"]
            or restored_metrics.get("height") != prior_metrics["height"]
        ):
            raise RuntimeError("CDP viewport did not restore after clearing override")
        if value is None:
            raise RuntimeError("CDP DOM evaluation returned no value")
        snapshot = json.loads(value)
        snapshot["viewport_restore"] = {
            "method": "Emulation.clearDeviceMetricsOverride",
            "verified": True,
            "prior": prior_metrics,
            "restored": restored_metrics,
        }
        return _canonical_json_bytes(snapshot)
    finally:
        stream.close()


def _v5_cdp_evaluate_json(
    cdp_base_url: str,
    expression: str,
    *,
    timeout: float = 10.0,
) -> Any:
    """Evaluate one read-only expression in the sole loopback Electron page."""

    parsed = urlsplit(cdp_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
    ):
        raise ValueError("CDP base URL must be loopback HTTP")
    targets = json.loads(
        _v5_loopback_http_get(
            f"http://{parsed.hostname}:{parsed.port}/json/list",
            timeout=timeout,
        )
    )
    pages = [
        item
        for item in targets
        if isinstance(item, dict)
        and item.get("type") == "page"
        and isinstance(item.get("webSocketDebuggerUrl"), str)
    ]
    if len(pages) != 1:
        raise RuntimeError("CDP did not expose exactly one page target")
    websocket_url = urlsplit(pages[0]["webSocketDebuggerUrl"])
    if (
        websocket_url.scheme != "ws"
        or websocket_url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or websocket_url.port is None
    ):
        raise RuntimeError("CDP websocket is not loopback")
    stream = socket.create_connection(
        (websocket_url.hostname, websocket_url.port),
        timeout=timeout,
    )
    try:
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {websocket_url.path} HTTP/1.1\r\n"
            f"Host: {websocket_url.hostname}:{websocket_url.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        stream.sendall(request)
        handshake = b""
        while b"\r\n\r\n" not in handshake:
            handshake += stream.recv(4096)
            if len(handshake) > 65536:
                raise RuntimeError("oversized CDP websocket handshake")
        if not handshake.startswith(b"HTTP/1.1 101"):
            raise RuntimeError("CDP websocket upgrade failed")
        command = json.dumps(
            {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            },
            separators=(",", ":"),
        ).encode()
        stream.sendall(_v5_websocket_frame(command))
        while True:
            opcode, payload = _v5_recv_websocket_frame(stream)
            if opcode == 0x9:
                stream.sendall(bytes((0x8A, len(payload))) + payload)
                continue
            if opcode == 0x8:
                raise RuntimeError("CDP websocket closed before response")
            if opcode != 0x1:
                continue
            decoded = json.loads(payload)
            if decoded.get("id") != 1:
                continue
            if decoded.get("error") or nested(
                decoded,
                "result",
                "exceptionDetails",
            ):
                raise RuntimeError("CDP Runtime.evaluate failed")
            value = nested(decoded, "result", "result", "value")
            if not isinstance(value, str):
                raise RuntimeError("CDP Runtime.evaluate returned no JSON")
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise RuntimeError("CDP Runtime.evaluate returned malformed JSON") from exc
    finally:
        stream.close()


def _v5_cdp_session_logs(
    cdp_base_url: str,
    session_id: str,
) -> list[str]:
    expression = (
        "(async()=>JSON.stringify(await (globalThis.api||window.api)"
        f".sessions.getLogs({json.dumps(session_id)})))()"
    )
    value = _v5_cdp_evaluate_json(cdp_base_url, expression)
    if not isinstance(value, list) or not all(isinstance(row, str) for row in value):
        raise RuntimeError("Electron session logs are not a string array")
    return value


def _v5_hash_python_runtime(imported_init: Path) -> dict[str, Any] | None:
    try:
        package_pin = _v5_pin_regular_file(imported_init)
        server_pin = _v5_pin_regular_file(imported_init.parent / "server.py")
    except (OSError, ValueError):
        return None
    digest = hashlib.sha256()
    count = 0
    try:
        paths = sorted(imported_init.parent.rglob("*.py"))
        for path in paths:
            pin = _v5_pin_regular_file(path)
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != pin["sha256"]:
                return None
            relative = (
                Path("vmlx_engine") / path.relative_to(imported_init.parent)
            ).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(content)
            digest.update(b"\0")
            count += 1
    except (OSError, ValueError):
        return None
    return {
        "package_init_path": package_pin["path"],
        "package_init_sha256": package_pin["sha256"],
        "server_module_path": server_pin["path"],
        "server_module_sha256": server_pin["sha256"],
        "python_source_tree_sha256": digest.hexdigest(),
        "python_source_file_count": count,
    }


def _v5_runtime_source_attestation(
    runtime: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate the path-free import-time source hashes published by /health."""

    if (
        runtime.get("package_init_relpath") != "vmlx_engine/__init__.py"
        or runtime.get("server_module_relpath") != "vmlx_engine/server.py"
        or not isinstance(runtime.get("python_source_file_count"), int)
        or runtime.get("python_source_file_count", 0) <= 0
        or runtime.get("python_source_read_error_count") != 0
    ):
        return None
    fields = (
        "package_init_sha256",
        "server_module_sha256",
        "python_source_tree_sha256",
        "python_executable_fingerprint_sha256",
    )
    if any(
        not isinstance(runtime.get(field), str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime[field]) is None
        for field in fields
    ):
        return None
    return {
        field: runtime[field]
        for field in (
            *fields,
            "python_source_file_count",
            "python_source_read_error_count",
        )
    }


def _v5_backend_invocation_fingerprint(
    backend: dict[str, Any],
) -> str | None:
    """Mirror the runtime's non-resolving sys.executable fingerprint."""

    argv = backend.get("argv")
    executable_path = backend.get("executable_path")
    if (
        not isinstance(argv, list)
        or not argv
        or not isinstance(argv[0], str)
        or not isinstance(executable_path, str)
    ):
        return None
    invocation = Path(argv[0])
    observed = Path(executable_path)
    try:
        if (
            not invocation.is_absolute()
            or not invocation.is_file()
            or invocation.resolve() != observed.resolve()
        ):
            return None
    except OSError:
        return None
    return hashlib.sha256(str(invocation.absolute()).encode()).hexdigest()


def _v5_validate_runtime_bundle_attestation(
    health: dict[str, Any],
    runtime: dict[str, Any],
    bundle_snapshot: dict[str, Any],
) -> bool:
    """Bind the held runtime to the production nested bundle attestation."""

    health_attestation = health.get("model_bundle_provenance")
    runtime_attestation = runtime.get("model_bundle_provenance")
    if (
        not isinstance(health_attestation, dict)
        or runtime_attestation != health_attestation
        or health_attestation.get("fingerprint_sha256")
        != bundle_snapshot.get("fingerprint_sha256")
    ):
        return False
    return (
        _validated_bundle_attestation(
            bundle_snapshot.get("model_bundle_path"),
            health_attestation,
        )
        is not None
    )


def _v5_independent_runtime_observation(
    args: argparse.Namespace,
    source: dict[str, Any],
    bundle_snapshot: dict[str, Any],
    hooks: dict[str, Any] | None,
) -> dict[str, Any]:
    provider = (hooks or {}).get("raw_runtime_observer")
    if callable(provider):
        raw = provider(args, source, bundle_snapshot)
    else:
        health_bytes = _v5_loopback_http_get(args.health_url)
        dom_bytes = _v5_cdp_dom_snapshot(args.cdp_url)
        raw = {
            "health_bytes": health_bytes,
            "dom_bytes": dom_bytes,
            "backend_pid": args.backend_pid,
            "gateway_pid": args.gateway_pid,
            "electron_pid": args.electron_pid,
            "direct_listener": _observe_listener(*_v5_host_port(args.direct_base_url)),
            "gateway_listener": _observe_listener(*_v5_host_port(args.gateway_base_url)),
        }
    health_bytes = raw.get("health_bytes")
    dom_bytes = raw.get("dom_bytes")
    if not isinstance(health_bytes, bytes) or not isinstance(dom_bytes, bytes):
        raise RuntimeError("runtime observer did not return raw health and DOM bytes")
    health = json.loads(health_bytes)
    dom = json.loads(dom_bytes)
    if not isinstance(health, dict) or not isinstance(dom, dict):
        raise RuntimeError("runtime health or DOM observation is not object JSON")
    runtime = health.get("runtime_provenance")
    if not isinstance(runtime, dict):
        raise RuntimeError("health has no runtime provenance")
    runtime_hashes = _v5_runtime_source_attestation(runtime)
    if runtime_hashes is None:
        raise RuntimeError("health runtime source provenance is incomplete")
    process_observer = (hooks or {}).get("process_observer", _observe_process)
    listener_observer = (hooks or {}).get("listener_observer")
    backend = process_observer(int(raw.get("backend_pid") or 0))
    gateway = process_observer(int(raw.get("gateway_pid") or 0))
    electron = process_observer(int(raw.get("electron_pid") or 0))
    if not all(isinstance(value, dict) for value in (backend, gateway, electron)):
        raise RuntimeError("runtime process observation is incomplete")
    direct_expected = {
        "host": _v5_host_port(args.direct_base_url)[0],
        "port": _v5_host_port(args.direct_base_url)[1],
        "owner_pid": backend["pid"],
    }
    gateway_expected = {
        "host": _v5_host_port(args.gateway_base_url)[0],
        "port": _v5_host_port(args.gateway_base_url)[1],
        "owner_pid": gateway["pid"],
    }
    if callable(listener_observer):
        direct_listener = listener_observer(*_v5_host_port(args.direct_base_url))
        gateway_listener = listener_observer(*_v5_host_port(args.gateway_base_url))
    else:
        direct_listener = raw.get("direct_listener")
        gateway_listener = raw.get("gateway_listener")
    if direct_listener != direct_expected or gateway_listener != gateway_expected:
        raise RuntimeError("runtime listeners do not match observed processes")
    source_attestation_observer = (hooks or {}).get("source_attestation_observer")
    expected = (
        source_attestation_observer()
        if callable(source_attestation_observer)
        else release_runtime_source_attestation()
    )
    if (
        runtime_hashes["package_init_sha256"] != expected["package_init_sha256"]
        or runtime_hashes["server_module_sha256"] != expected["server_module_sha256"]
        or runtime_hashes["python_source_tree_sha256"]
        != expected["python_source_tree_sha256"]
        or runtime_hashes["python_source_file_count"]
        != expected["python_source_file_count"]
        or runtime_hashes["python_source_read_error_count"] != 0
        or runtime_hashes["python_executable_fingerprint_sha256"]
        != _v5_backend_invocation_fingerprint(backend)
        or dom.get("sourceCommit") != source.get("commit")
        or not _v5_validate_runtime_bundle_attestation(
            health,
            runtime,
            bundle_snapshot,
        )
    ):
        raise RuntimeError("live runtime does not match exact source or bundle")
    return {
        "health_bytes_sha256": hashlib.sha256(health_bytes).hexdigest(),
        "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
        "health": health,
        "dom": dom,
        "backend": backend,
        "gateway": gateway,
        "electron": electron,
        "direct_listener": direct_listener,
        "gateway_listener": gateway_listener,
        "runtime_hashes": runtime_hashes,
        "expected_source_attestation": expected,
        "bundle_fingerprint_sha256": bundle_snapshot["fingerprint_sha256"],
    }


def _v5_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
    ):
        raise ValueError(f"invalid loopback base URL: {base_url}")
    return str(parsed.hostname), int(parsed.port)


def _v5_collect_owned_captures(
    producer_results: dict[str, dict[str, Any]],
    run_context: dict[str, Any],
) -> dict[str, list[tuple[dict[str, Any], bytes]]]:
    collected: dict[str, list[tuple[dict[str, Any], bytes]]] = {
        "ui": [],
        "api": [],
        "cache": [],
    }
    expected_schema = {
        "ui": V5_UI_SCHEMA,
        "api": V5_API_SCHEMA,
        "cache": V5_CACHE_SCHEMA,
    }
    for producer in V5_PRODUCER_NAMES:
        result = producer_results.get(producer)
        envelope = result.get("capture") if isinstance(result, dict) else None
        captures = envelope.get("captures") if isinstance(envelope, dict) else None
        if not isinstance(captures, list) or not captures:
            raise RuntimeError(f"{producer} producer returned no captures")
        for index, encoded in enumerate(captures):
            raw = _v5_decode_bytes(encoded)
            if raw is None:
                raise RuntimeError(f"{producer} capture[{index}] is not exact bytes")
            try:
                artifact = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{producer} capture[{index}] is not JSON"
                ) from exc
            if (
                not isinstance(artifact, dict)
                or artifact.get("schema") != expected_schema[producer]
                or artifact.get("run_id") != run_context["run_id"]
                or artifact.get("nonce") != run_context["nonce"]
            ):
                raise RuntimeError(f"{producer} capture[{index}] binding mismatch")
            collected[producer].append((artifact, raw))
    return collected


def _v5_json_bytes(encoded: Any) -> tuple[Any, bytes] | tuple[None, None]:
    raw = _v5_decode_bytes(encoded)
    if raw is None:
        return None, None
    try:
        return json.loads(raw), raw
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None


def _v5_jsonl_bytes(encoded: Any) -> tuple[list[dict[str, Any]], bytes] | tuple[None, None]:
    raw = _v5_decode_bytes(encoded)
    if raw is None:
        return None, None
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return None, None
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    return rows, raw


def _v5_has_bad_repetition(text: str) -> bool:
    words = re.findall(r"\S+", text)
    if len(words) < 12:
        return False
    for width in range(3, min(24, len(words) // 3) + 1):
        tail = words[-width:]
        if words[-2 * width : -width] == tail and words[-3 * width : -2 * width] == tail:
            return True
    return False


def _v5_ui_facts(
    captures: list[tuple[dict[str, Any], bytes]],
    runtime: dict[str, Any],
    bundle_snapshot: dict[str, Any],
) -> tuple[set[str], list[str]]:
    if len(captures) != len(V5_CACHE_PHASES):
        return set(), []
    artifacts_by_phase: dict[int, tuple[dict[str, Any], bytes]] = {}
    source_proofs_by_phase: dict[int, dict[str, Any]] = {}
    prefix_hashes: list[str] = []
    for artifact, raw_artifact in captures:
        phase_index = artifact.get("phase_index")
        if (
            not isinstance(phase_index, int)
            or phase_index in artifacts_by_phase
            or not 0 <= phase_index < len(V5_CACHE_PHASES)
        ):
            return set(), []
        phase = V5_CACHE_PHASES[phase_index]
        turns = artifact.get("turns")
        interaction, interaction_bytes = _v5_json_bytes(
            artifact.get("interaction_b64")
        )
        source_proof, source_proof_bytes = _v5_json_bytes(
            artifact.get("source_proof_b64")
        )
        session_id = str(artifact.get("session_id") or "")
        source_ui_turns = source_proof.get("uiTurnEvidence")
        cache_rows = source_proof.get("cacheRequestEvidence")
        cache_correlation = source_proof.get("cacheRequestCorrelation")
        if (
            artifact.get("phase_name") != phase["name"]
            or artifact.get("representative_id")
            != phase["representative_id"]
            or artifact.get("ui_action_profile")
            != phase["ui_action_profile"]
            or artifact.get("ui_turn_count") != phase["ui_turn_count"]
            or not isinstance(turns, list)
            or len(turns) != phase["ui_turn_count"]
            or not isinstance(interaction, list)
            or len(interaction) != 1
            or not isinstance(source_proof, dict)
            or source_proof.get("format") != ELECTRON_PROOF_SCHEMA
            or source_proof.get("run_id") != artifact.get("run_id")
            or nested(source_proof, "session", "id") != session_id
            or nested(source_proof, "uiStartControl", "clicked") is not True
            or not isinstance(source_ui_turns, list)
            or len(source_ui_turns) != phase["ui_turn_count"]
            or not isinstance(cache_rows, list)
            or len(cache_rows) != phase["ui_turn_count"]
            or not isinstance(cache_correlation, dict)
            or cache_correlation.get("status") != "verified"
            or not session_id
        ):
            return set(), []
        for ui_turn, cache_row in zip(
            source_ui_turns,
            cache_rows,
            strict=True,
        ):
            if not isinstance(ui_turn, dict) or not isinstance(cache_row, dict):
                return set(), []
            terminal_response_id = str(
                ui_turn.get("terminalResponseId") or ""
            )
            health = cache_row.get("healthAfter")
            execution = nested(health, "scheduler", "last_cache_execution")
            observation = cache_row.get("serverObservation")
            health_artifact = cache_row.get("healthArtifact")
            if (
                not terminal_response_id
                or cache_row.get("correlationStatus") != "verified"
                or cache_row.get("terminalResponseId")
                != terminal_response_id
                or cache_row.get("serverRequestId")
                != terminal_response_id
                or cache_row.get("executionRequestId")
                != terminal_response_id
                or not isinstance(execution, dict)
                or execution.get("request_id") != terminal_response_id
                or not isinstance(observation, dict)
                or observation.get("request_id") != terminal_response_id
                or observation.get("terminal_response_id")
                != terminal_response_id
                or observation.get("message_id")
                != ui_turn.get("assistantMessageId")
                or not isinstance(health_artifact, dict)
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(health_artifact.get("sha256") or ""),
                )
                or not isinstance(health_artifact.get("size_bytes"), int)
                or health_artifact["size_bytes"] <= 0
            ):
                return set(), []
            health_path = Path(str(health_artifact.get("path") or ""))
            try:
                health_pin = _v5_pin_regular_file(health_path)
            except (OSError, ValueError):
                return set(), []
            if (
                path_within(health_path, ROOT)
                or health_pin["sha256"] != health_artifact["sha256"]
                or health_pin["size"] != health_artifact["size_bytes"]
            ):
                return set(), []
            prefix_hashes.append(health_pin["sha256"])
        artifacts_by_phase[phase_index] = artifact, raw_artifact
        source_proofs_by_phase[phase_index] = source_proof
        prefix_hashes.extend(
            (
                hashlib.sha256(raw_artifact).hexdigest(),
                hashlib.sha256(interaction_bytes).hexdigest(),
                hashlib.sha256(source_proof_bytes).hexdigest(),
            )
        )
    if set(artifacts_by_phase) != set(range(len(V5_CACHE_PHASES))):
        return set(), []
    primary_sessions = {
        str(artifacts_by_phase[index][0].get("session_id") or "")
        for index in range(5)
    }
    native_session = str(
        artifacts_by_phase[5][0].get("session_id") or ""
    )
    if (
        len(primary_sessions) != 1
        or not native_session
        or native_session in primary_sessions
    ):
        return set(), []
    phase_observations = runtime.get("phase_observations")
    if not isinstance(phase_observations, list):
        return set(), []
    observation_by_phase = {
        row.get("phase_index"): row.get("observation")
        for row in phase_observations
        if isinstance(row, dict)
    }
    if set(observation_by_phase) != set(range(len(V5_CACHE_PHASES))):
        return set(), []
    artifact, raw_artifact = artifacts_by_phase[5]
    selected_runtime = observation_by_phase[5]
    if not isinstance(selected_runtime, dict):
        return set(), []
    turns = artifact.get("turns")
    interaction, interaction_bytes = _v5_json_bytes(artifact.get("interaction_b64"))
    source_proof, source_proof_bytes = _v5_json_bytes(
        artifact.get("source_proof_b64")
    )
    dom = selected_runtime.get("dom")
    if (
        not isinstance(turns, list)
        or len(turns) < 3
        or not isinstance(interaction, list)
        or not isinstance(source_proof, dict)
        or not isinstance(dom, dict)
    ):
        return set(), []
    session_id = str(artifact.get("session_id") or "")
    start_actions = [
        row
        for row in interaction
        if isinstance(row, dict)
        and row.get("method")
        in {
            "Input.dispatchMouseEvent",
            "Runtime.evaluate.visibleHTMLElement.click",
        }
        and row.get("selector") == "button[data-action='start-session']"
        and row.get("session_id") == session_id
    ]
    dom_messages = dom.get("messages")
    if (
        not session_id
        or len(start_actions) != 1
        or not isinstance(dom_messages, list)
        or source_proof.get("format") != ELECTRON_PROOF_SCHEMA
        or source_proof.get("run_id") != artifact.get("run_id")
        or nested(source_proof, "session", "id") != session_id
        or nested(source_proof, "uiStartControl", "clicked") is not True
    ):
        return set(), []
    hashes = prefix_hashes + [
        hashlib.sha256(raw_artifact).hexdigest(),
        hashlib.sha256(interaction_bytes).hexdigest(),
        hashlib.sha256(source_proof_bytes).hexdigest(),
        selected_runtime["dom_bytes_sha256"],
    ]
    facts: set[str] = {"real_start_button"}
    reasoning_values: list[str] = []
    content_values: list[str] = []
    assistant_records = source_proof.get("assistantRecords")
    persisted_reasoning = source_proof.get("persistedReasoningByMessage")
    cache_request_evidence = source_proof.get("cacheRequestEvidence")
    if (
        not isinstance(assistant_records, list)
        or len(assistant_records) < 3
        or not isinstance(persisted_reasoning, list)
        or len(persisted_reasoning) < 3
        or not isinstance(cache_request_evidence, list)
        or len(cache_request_evidence) < 3
    ):
        return set(), []
    saw_tool_call = False
    saw_tool_result = False
    timings_match = True
    rendering_requested = False
    rendering_match = True
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return set(), []
        request, request_bytes = _v5_json_bytes(turn.get("request_b64"))
        events, events_bytes = _v5_jsonl_bytes(turn.get("events_b64"))
        if (
            not isinstance(request, dict)
            or not isinstance(events, list)
            or not events
            or index >= len(dom_messages)
            or not isinstance(dom_messages[index], dict)
        ):
            return set(), []
        if [row.get("seq") for row in events] != list(range(len(events))):
            return set(), []
        if not _successful_terminal(events):
            return set(), []
        reasoning = "".join(
            str(row.get("text") or "")
            for row in events
            if row.get("type") == "reasoning_delta"
        )
        content = "".join(
            str(row.get("text") or "")
            for row in events
            if row.get("type") == "content_delta"
        )
        message = dom_messages[index]
        source_message = assistant_records[index]
        source_reasoning_rows = persisted_reasoning[index]
        source_cache_row = cache_request_evidence[index]
        if (
            not isinstance(source_message, dict)
            or not isinstance(source_reasoning_rows, list)
            or not isinstance(source_cache_row, dict)
        ):
            return set(), []
        source_reasoning = "\n".join(
            str(row.get("text") or "")
            for row in source_reasoning_rows
            if isinstance(row, dict) and str(row.get("text") or "")
        )
        expected = _expected_visible_final(request)
        terminal_response_id = next(
            (
                str(row.get("response_id") or "")
                for row in events
                if row.get("type") == "terminal"
            ),
            "",
        )
        if (
            not content.strip()
            or source_message.get("content") != content
            or source_reasoning != reasoning
            or not str(message.get("content_text") or "").strip()
            or (reasoning and not str(message.get("reasoning_text") or "").strip())
            or (
                message.get("terminal_text") is not None
                and message.get("terminal_text") != "Completed"
            )
            or (expected and content.strip() != expected)
            or CONTROL_MARKER_RE.search(content)
            or _v5_has_bad_repetition(content)
            or not terminal_response_id
            or source_cache_row.get("serverRequestId")
            != terminal_response_id
            or source_cache_row.get("executionRequestId")
            != terminal_response_id
        ):
            return set(), []
        event_timing = {
            "ttft_ms": next(
                (
                    row.get("ttft_ms")
                    for row in events
                    if row.get("type") == "terminal"
                ),
                None,
            ),
            "decode_tps": next(
                (
                    row.get("decode_tps")
                    for row in events
                    if row.get("type") == "terminal"
                ),
                None,
            ),
        }
        timings_match = timings_match and all(
            isinstance(event_timing[key], (int, float))
            and (
                message.get(key) is None
                or message.get(key) == event_timing[key]
            )
            for key in event_timing
        )
        html = str(message.get("html") or "")
        request_text = json.dumps(request, sort_keys=True)
        if "$43" in request_text or "\\\\(" in request_text:
            rendering_requested = True
            rendering_match = rendering_match and (
                "$43" in html
                and 'class="katex"' in html
                and "\\times" not in html
            )
        saw_tool_call = saw_tool_call or any(
            row.get("type") == "tool_call"
            and row.get("name") in {"file_info", "run_command"}
            and isinstance(row.get("arguments"), dict)
            for row in events
        )
        saw_tool_result = saw_tool_result or any(
            row.get("type") == "tool_result"
            and row.get("call_id")
            and str(row.get("content") or "")
            for row in events
        )
        reasoning_values.append(reasoning)
        content_values.append(content)
        hashes.extend(
            (
                hashlib.sha256(request_bytes).hexdigest(),
                hashlib.sha256(events_bytes).hexdigest(),
            )
        )
    facts.update(
        {
            "minimum_three_turns",
            "visible_content",
            "terminal_state",
            "nonempty_final",
            "no_control_markup",
        }
    )
    if sum(bool(value) for value in reasoning_values) >= 2:
        facts.add("reasoning_rail")
    if saw_tool_call and saw_tool_result:
        facts.update({"tool_result_continuation", "exact_tool_arguments"})
        if reasoning_values[0] and reasoning_values[1]:
            facts.add("reasoning_tool_reasoning_tool_answer")
    if timings_match:
        facts.add("cache_ttft_tps")
    if rendering_requested and rendering_match:
        facts.update(
            {
                "rendering",
                "katex_rendered",
                "currency_preserved",
                "markdown_rendered",
            }
        )
    if len(set(content_values)) == len(content_values):
        facts.update(
            {
                "coherence",
                "no_stale_reasoning_replay",
                "no_looping_or_gibberish",
            }
        )
    visible_text = str(dom.get("text") or "")
    locales = dom.get("locales")
    viewport = dom.get("viewport")
    viewport_restore = dom.get("viewport_restore")
    catalog_keys = dom.get("catalog_translation_keys")
    supported_locales = dom.get("supported_locales")
    picker_locales = dom.get("picker_supported_locales")
    visible_locale_options = dom.get("visible_locale_options")
    locale_records = (
        locales
        if isinstance(locales, list)
        and all(isinstance(row, dict) for row in locales)
        else []
    )
    catalog_key_set = {
        str(key)
        for key in catalog_keys
        if isinstance(key, str) and key
    } if isinstance(catalog_keys, list) else set()
    canonical_key_lookup = {key.lower() for key in catalog_key_set}
    visible_key_tokens = {
        token.lower()
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+",
            visible_text,
        )
    }
    locale_keys_clean = bool(locale_records) and all(
        row.get("selected_locale") == row.get("locale")
        and row.get("raw_translation_keys") == []
        for row in locale_records
    )
    locale_catalog_exact = (
        dom.get("locale_catalog_source") == "main_ipc_canonical_locale_json"
        and bool(catalog_key_set)
        and dom.get("translation_key_count") == len(catalog_key_set)
        and isinstance(supported_locales, list)
        and bool(supported_locales)
        and sorted(str(value) for value in supported_locales)
        == sorted(str(value) for value in picker_locales or [])
        == sorted(str(value) for value in visible_locale_options or [])
    )
    if (
        locale_catalog_exact
        and not (visible_key_tokens & canonical_key_lookup)
        and dom.get("raw_translation_keys") == []
        and locale_keys_clean
    ):
        facts.add("no_raw_translation_keys")
    observed_locales = (
        [str(row.get("locale") or "") for row in locale_records]
        if locale_records
        else locales
    )
    if (
        isinstance(observed_locales, list)
        and sorted(observed_locales)
        == sorted(str(value) for value in supported_locales or [])
        and locale_catalog_exact
        and locale_keys_clean
    ):
        facts.add("all_supported_locales_checked")
    if (
        isinstance(viewport, dict)
        and isinstance(viewport.get("width"), int)
        and viewport["width"] <= 640
        and isinstance(viewport.get("scroll_width"), int)
        and viewport["scroll_width"] <= viewport["width"]
        and isinstance(viewport_restore, dict)
        and viewport_restore.get("method")
        == "Emulation.clearDeviceMetricsOverride"
        and viewport_restore.get("verified") is True
        and nested(viewport_restore, "prior", "width")
        == nested(viewport_restore, "restored", "width")
        and nested(viewport_restore, "prior", "height")
        == nested(viewport_restore, "restored", "height")
    ):
        facts.add("minimum_window_width_checked")
    defaults = bundle_snapshot["derived"]["generation_defaults"]

    def canonical_defaults(value: Any) -> dict[str, int | float]:
        if not isinstance(value, dict):
            return {}
        aliases = {
            "temperature": ("temperature",),
            "top_p": ("top_p", "topP"),
            "top_k": ("top_k", "topK"),
            "min_p": ("min_p", "minP"),
            "repetition_penalty": (
                "repetition_penalty",
                "repeatPenalty",
            ),
            "max_output_tokens": (
                "max_output_tokens",
                "max_new_tokens",
                "max_tokens",
                "maxNewTokens",
                "maxTokens",
            ),
        }
        result: dict[str, int | float] = {}
        for target, keys in aliases.items():
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, (int, float)) and not isinstance(
                    candidate,
                    bool,
                ):
                    result[target] = candidate
                    break
        return result

    def matches_defaults(observed: Any, expected: dict[str, Any]) -> bool:
        normalized = canonical_defaults(observed)
        return bool(expected) and all(
            key in normalized
            and abs(float(normalized[key]) - float(value)) <= 1e-6
            for key, value in expected.items()
        )

    expected_defaults = canonical_defaults(defaults)
    initial_proof = source_proofs_by_phase.get(0) or {}
    live_contract_defaults = nested(
        initial_proof,
        "bundleGenerationContract",
        "defaults",
    )
    renderer_defaults = initial_proof.get("rendererGenerationDefaults")
    visible_settings = initial_proof.get("chatSettingsDom")
    visible_values = (
        visible_settings.get("values")
        if isinstance(visible_settings, dict)
        else None
    )
    health_defaults = nested(initial_proof, "server", "health", "effective_defaults")
    resolved_sampling_records = initial_proof.get("resolvedSamplingRecords")

    def exact_request_correlation(proof: dict[str, Any]) -> bool:
        expected_turns = int(nested(proof, "requestContract", "uiTurnCount") or 0)
        ui_turns = proof.get("uiTurnEvidence")
        correlation = proof.get("requestCorrelation")
        records = proof.get("resolvedSamplingRecords")
        if (
            expected_turns <= 0
            or not isinstance(ui_turns, list)
            or len(ui_turns) != expected_turns
            or not isinstance(correlation, dict)
            or correlation.get("status") != "verified"
            or not isinstance(correlation.get("turns"), list)
            or len(correlation["turns"]) != expected_turns
            or not isinstance(records, list)
        ):
            return False
        all_request_ids: list[str] = []
        for ui_turn in ui_turns:
            if not isinstance(ui_turn, dict):
                return False
            proof_id = str(ui_turn.get("proofRequestId") or "")
            user_id = str(ui_turn.get("userMessageId") or "")
            assistant_id = str(ui_turn.get("assistantMessageId") or "")
            request_ids = [
                str(value)
                for value in ui_turn.get("requestIds", [])
                if str(value)
            ] if isinstance(ui_turn.get("requestIds"), list) else []
            row = next(
                (
                    candidate
                    for candidate in correlation["turns"]
                    if isinstance(candidate, dict)
                    and candidate.get("turn") == ui_turn.get("turn")
                ),
                None,
            )
            if (
                not proof_id
                or proof_id != user_id
                or not assistant_id
                or ui_turn.get("terminalProofRequestId") != proof_id
                or ui_turn.get("terminalMessageId") != assistant_id
                or ui_turn.get("logMatchMode") != "exact_identity_ring_safe"
                or not request_ids
                or len(request_ids) != len(set(request_ids))
                or not isinstance(row, dict)
                or row.get("proofRequestId") != proof_id
                or row.get("userMessageId") != user_id
                or row.get("assistantMessageId") != assistant_id
                or row.get("serverProofRequestId") != proof_id
                or row.get("serverMessageId") != assistant_id
                or row.get("resolvedLogCorrelated") is not True
                or row.get("serverRequestIds") != request_ids
            ):
                return False
            matching = [
                record
                for record in records
                if isinstance(record, dict)
                and record.get("proof_request_id") == proof_id
                and record.get("message_id") == assistant_id
            ]
            record_ids = [str(record.get("request_id") or "") for record in matching]
            if (
                len(matching) != len(request_ids)
                or any(not value for value in record_ids)
                or len(record_ids) != len(set(record_ids))
                or set(record_ids) != set(request_ids)
                or any(
                    record.get("correlation_source") != "server_emitted"
                    for record in matching
                )
            ):
                return False
            all_request_ids.extend(request_ids)
        return (
            len(all_request_ids) == len(set(all_request_ids))
            and len(all_request_ids) == len(records)
        )

    request_correlation_exact = exact_request_correlation(initial_proof)
    resolved_defaults_match = (
        request_correlation_exact
        and
        isinstance(resolved_sampling_records, list)
        and len(resolved_sampling_records)
        >= int(nested(initial_proof, "requestContract", "uiTurnCount") or 0)
        and all(
            isinstance(row, dict)
            and matches_defaults(row.get("values"), expected_defaults)
            for row in resolved_sampling_records
        )
    )
    request_sampling = nested(
        initial_proof,
        "requestContract",
        "samplingOverrides",
    )
    stored_overrides = initial_proof.get("chatOverrides")
    stored_sampling_absent = isinstance(stored_overrides, dict) and all(
        stored_overrides.get(key) is None
        for key in (
            "temperature",
            "topP",
            "topK",
            "minP",
            "repeatPenalty",
            "maxTokens",
        )
    )
    expected_without_output = {
        key: value
        for key, value in expected_defaults.items()
        if key != "max_output_tokens"
    }
    max_output = expected_defaults.get("max_output_tokens")
    max_tokens_visible = (
        isinstance(visible_settings, dict)
        and isinstance(visible_settings.get("maxTokens"), dict)
        and str(visible_settings["maxTokens"].get("value") or "") == ""
        and (
            max_output is None
            or str(max_output)
            in str(visible_settings["maxTokens"].get("placeholder") or "")
        )
    )
    if (
        matches_defaults(live_contract_defaults, expected_defaults)
        and matches_defaults(renderer_defaults, expected_defaults)
        and matches_defaults(visible_values, expected_without_output)
        and matches_defaults(health_defaults, expected_defaults)
        and resolved_defaults_match
        and max_tokens_visible
        and request_sampling == {}
        and stored_sampling_absent
    ):
        facts.add("bundle_defaults_in_new_ui_session")

    settings_interaction = initial_proof.get("chatSettingsInteraction")
    if (
        isinstance(settings_interaction, dict)
        and settings_interaction.get("openedVisibly") is True
        and settings_interaction.get("savedViaVisibleControl") is True
        and settings_interaction.get("reopenedAfterSave") is True
        and settings_interaction.get("persistedAfterReopen") is True
        and initial_proof.get("requestedBuiltinTools") is True
        and isinstance(stored_overrides, dict)
        and stored_overrides.get("builtinToolsEnabled") is True
        and stored_overrides.get("wireApi")
        == nested(initial_proof, "chatSettingsDom", "wireApi")
        and stored_overrides.get("workingDirectory")
        == initial_proof.get("workingDirectory")
    ):
        facts.add("ui_override_session_scoped")

    restarted_proof = source_proofs_by_phase.get(1) or {}
    initial_config = nested(initial_proof, "session", "effective_config")
    restarted_config = nested(restarted_proof, "session", "effective_config")
    persisted_keys = (
        "enablePrefixCache",
        "usePagedCache",
        "enableBlockDiskCache",
        "kvCacheQuantization",
        "blockDiskMaxSizeGB",
        "prefixCacheMemoryPercent",
    )
    persisted_values = {
        key: initial_config.get(key)
        for key in persisted_keys
        if isinstance(initial_config, dict) and key in initial_config
    }
    if (
        persisted_values
        and isinstance(restarted_config, dict)
        and all(restarted_config.get(key) == value for key, value in persisted_values.items())
        and nested(initial_proof, "serverCacheControls", "verified") is True
        and nested(restarted_proof, "serverCacheControls", "verified") is True
    ):
        facts.add("ui_override_restart_persisted")

    max_prompt_tokens = nested(initial_proof, "server", "health", "max_prompt_tokens")
    max_output_tokens = nested(
        initial_proof,
        "server",
        "health",
        "effective_defaults",
        "max_output_tokens",
    )
    if (
        isinstance(max_prompt_tokens, int)
        and max_prompt_tokens > 0
        and isinstance(max_output_tokens, int)
        and max_output_tokens > 0
        and max_prompt_tokens != max_output_tokens
    ):
        facts.add("max_context_output_distinct")
    cache_controls = initial_proof.get("serverCacheControls")
    cache_visible = (
        cache_controls.get("initialCacheControls")
        if isinstance(cache_controls, dict)
        else None
    )
    cache_config = (
        cache_controls.get("persistedConfig")
        if isinstance(cache_controls, dict)
        else None
    )
    cache_health = (
        cache_controls.get("healthNativeCache")
        if isinstance(cache_controls, dict)
        else None
    )
    cache_argv = (
        cache_controls.get("argv")
        if isinstance(cache_controls, dict)
        else None
    )
    cache_parity = bool(
        isinstance(cache_visible, dict)
        and isinstance(cache_config, dict)
        and isinstance(cache_health, dict)
        and isinstance(cache_argv, list)
        and cache_visible.get("enablePrefixCache")
        == cache_config.get("enablePrefixCache")
        and cache_visible.get("usePagedCache")
        == cache_config.get("usePagedCache")
        and cache_visible.get("enableBlockDiskCache")
        == cache_config.get("enableBlockDiskCache")
        and cache_config.get("enableBlockDiskCache") is True
        and "--enable-block-disk-cache" in cache_argv
        and cache_health.get("block_disk_l2") is True
        and cache_health.get("prefix")
        == cache_config.get("enablePrefixCache")
        and cache_health.get("paged") == cache_config.get("usePagedCache")
        and cache_health.get("block_disk_only")
        == (not cache_config.get("usePagedCache"))
        and (
            "--use-paged-cache"
            if cache_config.get("usePagedCache")
            else "--no-paged-cache"
        )
        in cache_argv
    )
    if (
        nested(initial_proof, "serverCacheControls", "verified") is True
        and cache_parity
    ):
        facts.add("preview_argv_health_parity")
    media = dom.get("media")
    if isinstance(media, list):
        types = {
            str(row.get("type") or "")
            for row in media
            if isinstance(row, dict)
            and row.get("request_sha256")
            and row.get("response_text")
        }
        if {"image", "video", "audio"} <= types:
            facts.update(
                {
                    "electron_attachment_flow",
                    "image",
                    "video",
                    "audio",
                }
            )
        salts = [
            row.get("cache_salt")
            for row in media
            if isinstance(row, dict) and row.get("cache_salt")
        ]
        if len(salts) >= 3 and salts[0] == salts[2] and salts[0] != salts[1]:
            facts.add("media_salt_isolation")
        if any(
            isinstance(row, dict) and row.get("post_media_tool_call")
            for row in media
        ):
            facts.add("post_media_tool_turn")
    return facts, hashes


def _v5_api_facts(
    captures: list[tuple[dict[str, Any], bytes]],
    bundle_snapshot: dict[str, Any],
) -> tuple[dict[str, set[str]], set[str], list[str]]:
    if len(captures) != len(V5_CACHE_PHASES):
        return {}, set(), []
    artifacts_by_phase: dict[int, tuple[dict[str, Any], bytes]] = {}
    grouped: dict[
        tuple[int, str, str, str],
        list[tuple[dict[str, Any], bytes, str]],
    ] = {}
    hashes: list[str] = []
    endpoints = {
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
        "ollama": "/api/chat",
    }
    for artifact, raw_artifact in captures:
        phase_index = artifact.get("phase_index")
        if (
            not isinstance(phase_index, int)
            or phase_index in artifacts_by_phase
            or not 0 <= phase_index < len(V5_CACHE_PHASES)
        ):
            return {}, set(), []
        phase = V5_CACHE_PHASES[phase_index]
        flows = artifact.get("flows")
        if (
            artifact.get("phase_name") != phase["name"]
            or artifact.get("representative_id")
            != phase["representative_id"]
            or artifact.get("api_action_profile")
            != phase["api_action_profile"]
            or not isinstance(flows, list)
        ):
            return {}, set(), []
        artifacts_by_phase[phase_index] = artifact, raw_artifact
        hashes.append(hashlib.sha256(raw_artifact).hexdigest())
        for flow in flows:
            if not isinstance(flow, dict):
                return {}, set(), []
            request, request_bytes = _v5_json_bytes(flow.get("request_b64"))
            response_bytes = _v5_decode_bytes(flow.get("response_b64"))
            protocol = str(flow.get("protocol") or "")
            route = str(flow.get("route") or "")
            mode = str(flow.get("mode") or "")
            reasoning_mode = str(flow.get("reasoning_mode") or "")
            parsed_response = (
                _parse_raw_protocol_stream_v5(protocol, response_bytes)
                if response_bytes is not None and mode == "stream"
                else _parse_raw_protocol_nonstream_v5(
                    protocol,
                    response_bytes,
                )
                if response_bytes is not None and mode == "nonstream"
                else None
            )
            if (
                protocol not in endpoints
                or route not in {"direct", "gateway"}
                or mode not in {"stream", "nonstream"}
                or reasoning_mode not in {"auto", "on", "off"}
                or flow.get("endpoint") != endpoints[protocol]
                or not isinstance(request, dict)
                or response_bytes is None
                or parsed_response is None
            ):
                return {}, set(), []
            timing, timing_bytes = _v5_json_bytes(flow.get("timing_b64"))
            if not isinstance(timing, dict):
                return {}, set(), []
            started_ns = timing.get("started_ns")
            first_byte_ns = timing.get("first_byte_ns")
            ended_ns = timing.get("ended_ns")
            output_tokens = timing.get("output_tokens")
            if (
                not all(
                    isinstance(value, int)
                    for value in (
                        started_ns,
                        first_byte_ns,
                        ended_ns,
                        output_tokens,
                    )
                )
                or not (started_ns < first_byte_ns <= ended_ns)
                or output_tokens <= 0
                or abs(
                    float(timing.get("displayed_ttft_ms") or -1)
                    - (first_byte_ns - started_ns) / 1_000_000
                )
                > 0.5
                or abs(
                    float(timing.get("displayed_tps") or -1)
                    - output_tokens
                    / ((ended_ns - first_byte_ns) / 1_000_000_000)
                )
                > 0.1
            ):
                return {}, set(), []
            reasoning_key = (
                "think" if protocol == "ollama" else "enable_thinking"
            )
            if (
                (reasoning_mode == "auto" and reasoning_key in request)
                or (
                    reasoning_mode == "on"
                    and request.get(reasoning_key) is not True
                )
                or (
                    reasoning_mode == "off"
                    and request.get(reasoning_key) is not False
                )
            ):
                return {}, set(), []
            grouped.setdefault(
                (phase_index, protocol, route, mode),
                [],
            ).append((request, response_bytes, reasoning_mode))
            hashes.extend(
                (
                    hashlib.sha256(request_bytes).hexdigest(),
                    hashlib.sha256(response_bytes).hexdigest(),
                    hashlib.sha256(timing_bytes).hexdigest(),
                )
            )
    if set(artifacts_by_phase) != set(range(len(V5_CACHE_PHASES))):
        return {}, set(), []
    protocol_facts: dict[str, set[str]] = {}
    for protocol in endpoints:
        by_route: list[set[str]] = []
        for phase_index in (0, 5):
            for mode in ("stream", "nonstream"):
                if any(
                    len(
                        grouped.get(
                            (phase_index, protocol, route, mode),
                            [],
                        )
                    )
                    != 3
                    for route in ("direct", "gateway")
                ):
                    return {}, set(), []
            for route in ("direct", "gateway"):
                rows = grouped[(phase_index, protocol, route, "stream")]
                by_route.append(
                    _api_flow_facts_from_raw(
                        protocol,
                        [value[0] for value in rows],
                        [value[1] for value in rows],
                    )
                )
        protocol_facts[protocol] = set.intersection(*by_route)
    for phase_index in (1, 2, 3, 4):
        if any(
            len(grouped.get((phase_index, "chat", route, "stream"), []))
            != 3
            for route in ("direct", "gateway")
        ):
            return {}, set(), []
    global_facts = set.intersection(*protocol_facts.values())
    global_facts.add("stream_and_nonstream")
    if all(
        not _v5_has_bad_repetition(
            str(_parse_raw_protocol_stream_v5(protocol, response)["content"])
        )
        for (_phase, protocol, _route, mode), rows in grouped.items()
        if mode == "stream"
        for _request, response, _reasoning_mode in rows
    ):
        global_facts.update(
            {
                "no_looping_or_gibberish",
                "raw_timing_matches",
                "success_finish_reasons",
            }
        )
    artifact = artifacts_by_phase[0][0]
    lifecycle, lifecycle_bytes = _v5_json_bytes(artifact.get("lifecycle_b64"))
    if isinstance(lifecycle, list):
        hashes.append(hashlib.sha256(lifecycle_bytes).hexdigest())
        transitions = [
            (
                row.get("event"),
                row.get("before"),
                row.get("after"),
            )
            for row in lifecycle
            if isinstance(row, dict)
        ]
        expected = {
            ("swap", "model_a_loaded", "model_b_loaded"),
            ("start", "session_created", "model_loaded"),
            ("disconnect", "streaming", "cancelled"),
            ("port_conflict", "bind_failed", "recovered"),
            ("lan_rollback", "lan_enabled", "loopback_only"),
            ("swap_soak", "iteration_1", "iteration_10"),
        }
        if expected <= set(transitions):
            global_facts.update(
                {
                    "one_model_only_swap",
                    "eager_load_on_start",
                    "stop_and_disconnect",
                    "port_conflict_recovery",
                    "lan_rollback",
                    "repeated_swap_soak",
                }
            )
    modes = {
        reasoning_mode
        for rows in grouped.values()
        for _request, _response, reasoning_mode in rows
    }
    if modes == {"auto", "on", "off"}:
        global_facts.add("auto_on_off_policy")
    if all("no_control_markup" in facts for facts in protocol_facts.values()):
        global_facts.update(
            {
                "reasoning_parsers",
                "tool_parsers",
                "json_xml_boundaries",
                "no_inline_think_leak",
            }
        )
    sampling, defaults_bytes = _v5_json_bytes(artifact.get("sampling_b64"))
    if isinstance(sampling, dict):
        hashes.append(hashlib.sha256(defaults_bytes).hexdigest())
        bundle_defaults = bundle_snapshot["derived"]["generation_defaults"]
        sampler_keys = (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
        )

        def numeric_mapping(
            value: Any,
            keys: tuple[str, ...],
        ) -> dict[str, int | float]:
            if not isinstance(value, dict):
                return {}
            return {
                key: candidate
                for key in keys
                if isinstance((candidate := value.get(key)), (int, float))
                and not isinstance(candidate, bool)
            }

        def values_match(
            observed: Any,
            expected: dict[str, int | float],
        ) -> bool:
            actual = numeric_mapping(observed, tuple(expected))
            return bool(expected) and set(actual) == set(expected) and all(
                abs(float(actual[key]) - float(value)) <= 1e-6
                for key, value in expected.items()
            )

        def values_include(
            observed: Any,
            expected: dict[str, int | float],
        ) -> bool:
            actual = numeric_mapping(observed, sampler_keys)
            return bool(expected) and all(
                key in actual
                and abs(float(actual[key]) - float(value)) <= 1e-6
                for key, value in expected.items()
            )

        observations = sampling.get("observations")
        validated_observations: list[
            tuple[dict[str, Any], dict[str, Any]]
        ] = []
        if (
            sampling.get("schema") == "vmlx-r18-owned-sampling-attestation-v1"
            and isinstance(observations, list)
            and len(observations) == 3
        ):
            expected_labels = ("default", "override", "after_override")
            seen_ids: set[str] = set()
            for expected_label, observation in zip(
                expected_labels,
                observations,
                strict=True,
            ):
                if not isinstance(observation, dict):
                    validated_observations = []
                    break
                request, request_bytes = _v5_json_bytes(
                    observation.get("request_b64")
                )
                result_bytes = _v5_decode_bytes(observation.get("result_b64"))
                resolved = observation.get("resolved")
                line_bytes = _v5_decode_bytes(
                    resolved.get("line_b64")
                    if isinstance(resolved, dict)
                    else None
                )
                proof_request_id = str(
                    observation.get("proof_request_id") or ""
                )
                request_id = str(observation.get("request_id") or "")
                message_id = str(observation.get("message_id") or "")
                identity_values = (
                    proof_request_id,
                    request_id,
                    message_id,
                )
                if (
                    observation.get("label") != expected_label
                    or not isinstance(request, dict)
                    or request_bytes is None
                    or result_bytes is None
                    or line_bytes is None
                    or not isinstance(resolved, dict)
                    or not all(
                        re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value)
                        for value in identity_values
                    )
                    or any(value in seen_ids for value in identity_values)
                    or observation.get("request_sha256")
                    != hashlib.sha256(request_bytes).hexdigest()
                    or observation.get("result_sha256")
                    != hashlib.sha256(result_bytes).hexdigest()
                    or resolved.get("line_sha256")
                    != hashlib.sha256(line_bytes).hexdigest()
                    or request.get("max_tokens") != 2
                    or request.get("enable_thinking") is not False
                    or resolved.get("values", {}).get("max_tokens") != 2
                    or resolved.get("values", {}).get("enable_thinking")
                    is not False
                ):
                    validated_observations = []
                    break
                try:
                    line = line_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    validated_observations = []
                    break
                reparsed = _v5_parse_resolved_sampling_log(
                    line,
                    route="/v1/chat/completions",
                    expected_models={str(request.get("model") or "")},
                    proof_request_id=proof_request_id,
                    request_id=request_id,
                    message_id=message_id,
                )
                if reparsed != resolved:
                    validated_observations = []
                    break
                seen_ids.update(identity_values)
                validated_observations.append((observation, request))

        if len(validated_observations) == 3:
            default_resolved = sampling.get("default_resolved")
            override_request = sampling.get("override_request")
            override_resolved = sampling.get("override_resolved")
            after_override = sampling.get("after_override_resolved")
            resolved_rows = [
                row[0]["resolved"]["values"]
                for row in validated_observations
            ]
            top_level_bound = (
                default_resolved == resolved_rows[0]
                and override_resolved == resolved_rows[1]
                and after_override == resolved_rows[2]
            )
            expected_sampler = numeric_mapping(bundle_defaults, sampler_keys)
            health_defaults = sampling.get("health_effective_defaults")
            expected_output = next(
                (
                    bundle_defaults[key]
                    for key in ("max_output_tokens", "max_new_tokens")
                    if isinstance(bundle_defaults.get(key), (int, float))
                    and not isinstance(bundle_defaults.get(key), bool)
                ),
                None,
            )
            health_output = (
                health_defaults.get("max_output_tokens")
                if isinstance(health_defaults, dict)
                else None
            )
            output_default_matches = (
                expected_output is None
                or (
                    isinstance(health_output, (int, float))
                    and not isinstance(health_output, bool)
                    and abs(float(health_output) - float(expected_output)) <= 1e-6
                )
            )
            if (
                top_level_bound
                and values_match(default_resolved, expected_sampler)
                and values_match(after_override, expected_sampler)
                and values_match(health_defaults, expected_sampler)
                and output_default_matches
                and all(
                    not any(key in request for key in sampler_keys)
                    for _observation, request in (
                        validated_observations[0],
                        validated_observations[2],
                    )
                )
            ):
                global_facts.add("bundle_defaults_in_api")
            expected_override = numeric_mapping(override_request, sampler_keys)
            override_payload = validated_observations[1][1]
            default_without_override = {
                key: value
                for key, value in expected_sampler.items()
                if key not in expected_override
            }
            if (
                top_level_bound
                and expected_override
                and set(expected_override) <= set(sampler_keys)
                and all(
                    key in override_payload
                    and abs(float(override_payload[key]) - float(value)) <= 1e-6
                    for key, value in expected_override.items()
                )
                and values_include(override_resolved, expected_override)
                and all(
                    key in numeric_mapping(override_resolved, sampler_keys)
                    and abs(
                        float(numeric_mapping(override_resolved, sampler_keys)[key])
                        - float(value)
                    )
                    <= 1e-6
                    for key, value in default_without_override.items()
                )
                and values_match(after_override, expected_sampler)
            ):
                global_facts.add("api_request_override_request_scoped")
    media_types: set[str] = set()
    for rows in grouped.values():
        for request, _response, _reasoning_mode in rows:
            request_text = json.dumps(request, sort_keys=True).lower()
            for media_type in ("image", "video", "audio"):
                if f"{media_type}/" in request_text or f"{media_type}_url" in request_text:
                    media_types.add(media_type)
    if {"image", "video", "audio"} <= media_types:
        global_facts.add("raw_api_flow")
    return protocol_facts, global_facts, hashes


def _v5_cache_facts(
    captures: list[tuple[dict[str, Any], bytes]],
    representatives: dict[str, dict[str, Any]],
) -> tuple[set[str], list[str]]:
    if (
        len(captures) != 1
        or set(representatives) != set(V5_REPRESENTATIVE_IDS)
    ):
        return set(), []
    artifact, raw_artifact = captures[0]
    scenarios = artifact.get("phases")
    if not isinstance(scenarios, list) or len(scenarios) != len(
        V5_CACHE_PHASES
    ):
        return set(), []
    facts: set[str] = set()
    hashes = [hashlib.sha256(raw_artifact).hexdigest()]
    paged_modes: set[bool] = set()
    store_summary_hashes: dict[tuple[str, bool, str], str] = {}
    session_ids: dict[str, set[str]] = {
        representative_id: set()
        for representative_id in V5_REPRESENTATIVE_IDS
    }
    q4_restart_pids: dict[bool, list[int]] = {False: [], True: []}
    all_backend_pids: list[int] = []
    q4_phase_indexes: set[int] = set()
    policy_roles: set[tuple[str, str]] = set()
    phase_summaries: dict[int, dict[str, Any]] = {}
    phase_summary_hashes: dict[int, str] = {}
    phase_executions: dict[int, dict[str, Any]] = {}
    phase_instantiated: dict[int, dict[str, Any]] = {}
    saw_phase2_ram_eviction = False
    saw_phase2_disk_eviction = False

    def integer(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    for expected_phase, scenario in zip(
        V5_CACHE_PHASES,
        scenarios,
        strict=True,
    ):
        gate_operation = _v5_cache_gate_operation(expected_phase)
        if (
            not isinstance(scenario, dict)
            or scenario.get("phase_index") != expected_phase["index"]
            or scenario.get("phase_name") != expected_phase["name"]
            or scenario.get("representative_id")
            != expected_phase["representative_id"]
            or scenario.get("bundle_role") != expected_phase["bundle_role"]
            or scenario.get("cache_policy") != expected_phase["cache_policy"]
            or scenario.get("kv_cache_quantization")
            != expected_phase["kv_cache_quantization"]
            or scenario.get("tq_policy") != expected_phase["tq_policy"]
            or scenario.get("session_policy")
            != expected_phase["session_policy"]
            or scenario.get("operation") != expected_phase["operation"]
            or scenario.get("ui_action_profile")
            != expected_phase["ui_action_profile"]
            or scenario.get("ui_turn_count")
            != expected_phase["ui_turn_count"]
            or scenario.get("api_action_profile")
            != expected_phase["api_action_profile"]
            or scenario.get("paged_ram") != expected_phase["paged_ram"]
        ):
            return set(), []
        representative = representatives.get(expected_phase["representative_id"])
        bundle_snapshot = (
            representative.get("bundle")
            if isinstance(representative, dict)
            else None
        )
        expected_model = (
            str(representative.get("model") or "")
            if isinstance(representative, dict)
            else ""
        )
        if not isinstance(bundle_snapshot, dict) or not expected_model:
            return set(), []
        summary, summary_bytes = _v5_json_bytes(scenario.get("summary_b64"))
        artifact_manifest, manifest_bytes = _v5_json_bytes(
            scenario.get("artifact_manifest_b64")
        )
        if (
            not isinstance(summary, dict)
            or not isinstance(artifact_manifest, list)
            or not artifact_manifest
            or any(
                not isinstance(row, dict)
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(row.get("sha256") or ""),
                )
                or integer(row.get("size")) <= 0
                for row in artifact_manifest
            )
        ):
            return set(), []
        identity = summary.get("identity")
        observed_engine = (
            identity.get("observed_engine")
            if isinstance(identity, dict)
            else None
        )
        observed_bundle = (
            identity.get("model_bundle_provenance")
            if isinstance(identity, dict)
            else None
        )
        summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
        summary_manifest_rows = [
            row
            for row in artifact_manifest
            if isinstance(row, dict)
            and row.get("relative_path") == "summary.json"
        ]
        paged = scenario.get("paged_ram")
        backend_pid = scenario.get("backend_pid")
        session_id = str(scenario.get("session_id") or "")
        if (
            summary.get("schema") != CACHE_PROOF_SCHEMA
            or summary.get("phase") != gate_operation
            or summary.get("model") != expected_model
            or scenario.get("model") != expected_model
            or scenario.get("bundle_fingerprint_sha256")
            != bundle_snapshot["fingerprint_sha256"]
            or summary.get("gate_ok") is not True
            or not isinstance(observed_engine, dict)
            or observed_engine.get("pid") != backend_pid
            or not isinstance(observed_bundle, dict)
            or observed_bundle.get("fingerprint_sha256")
            != bundle_snapshot["fingerprint_sha256"]
            or len(summary_manifest_rows) != 1
            or summary_manifest_rows[0].get("sha256") != summary_sha256
            or integer(summary_manifest_rows[0].get("size"))
            != len(summary_bytes)
            or not isinstance(paged, bool)
            or _v5_paged_mode_from_cache_summary(summary) != paged
            or not isinstance(backend_pid, int)
            or backend_pid <= 0
            or not session_id
        ):
            return set(), []
        representative_id = expected_phase["representative_id"]
        phase_summaries[expected_phase["index"]] = summary
        phase_summary_hashes[expected_phase["index"]] = summary_sha256
        cache_policy = expected_phase["cache_policy"]
        session_ids[representative_id].add(session_id)
        all_backend_pids.append(backend_pid)
        policy_roles.add((representative_id, cache_policy))
        if cache_policy == "q4":
            q4_restart_pids[paged].append(backend_pid)
        requests = summary.get("requests")
        if not isinstance(requests, list):
            return set(), []
        by_tag = {
            str(row.get("tag") or ""): row
            for row in requests
            if isinstance(row, dict)
        }
        partial_tag = (
            "partial_b"
            if gate_operation == "store"
            else "restart_partial_c"
        )
        partial = by_tag.get(partial_tag)
        execution = (
            partial.get("last_cache_execution")
            if isinstance(partial, dict)
            else None
        )
        token_contract = summary.get("tokenizer_lcp_contract")
        lcp_rows = (
            token_contract.get("longest_common_prefix_tokens")
            if isinstance(token_contract, dict)
            else None
        )
        lcp_key = (
            "A:B"
            if gate_operation == "store"
            else "A:C"
        )
        independent_lcp = (
            integer(lcp_rows.get(lcp_key))
            if isinstance(lcp_rows, dict)
            else 0
        )
        cached_tokens = (
            integer(execution.get("cached_tokens"))
            if isinstance(execution, dict)
            else 0
        )
        prompt_tokens = (
            integer(execution.get("prompt_tokens"))
            if isinstance(execution, dict)
            else 0
        )
        uncached_tokens = (
            integer(execution.get("uncached_prompt_tokens"))
            if isinstance(execution, dict)
            else 0
        )
        if (
            not isinstance(partial, dict)
            or partial.get("cache_contract_ok") is not True
            or independent_lcp <= 1
            or cached_tokens <= 0
            or cached_tokens > independent_lcp
            or prompt_tokens <= independent_lcp
            or uncached_tokens != prompt_tokens - cached_tokens
            or uncached_tokens <= 0
        ):
            return set(), []
        phase_executions[expected_phase["index"]] = execution
        if expected_phase["index"] == 2 and (
            integer(execution.get("disk_blocks")) <= 0
            or "disk"
            not in str(execution.get("cache_detail") or "").lower()
        ):
            return set(), []
        if cache_policy == "q4":
            facts.update(
                {
                    "ssd_l2_enabled",
                    "longest_prefix_partial_block_hit",
                    "uncached_suffix_prefilled",
                    "prefill_skip_measured",
                }
            )
            paged_modes.add(paged)
            if paged:
                facts.add("paged_ram_enabled")
            else:
                facts.add("paged_ram_disabled")
        topology = nested(
            summary,
            "identity",
            "cache_topology_provenance",
            "configuration",
        )
        instantiated = (
            topology.get("instantiated")
            if isinstance(topology, dict)
            else None
        )
        if not isinstance(instantiated, dict) or instantiated.get(
            "block_disk_l2"
        ) is not True:
            return set(), []
        store_key = (representative_id, bool(paged), cache_policy)
        if gate_operation == "store":
            warm = by_tag.get("warm_a")
            if (
                isinstance(warm, dict)
                and warm.get("cache_contract_ok") is True
                and integer(warm.get("cached_tokens")) > 0
                and cache_policy == "q4"
            ):
                facts.add("cross_chat_reuse")
                if paged:
                    facts.add("ram_blocks_filled")
            store_summary_hashes[store_key] = summary_sha256
        else:
            if (
                summary.get("probe_linkage_ok") is not True
                or scenario.get("linked_store_summary_sha256")
                != store_summary_hashes.get(store_key)
                or integer(
                    execution.get("disk_blocks")
                    if isinstance(execution, dict)
                    else None
                )
                <= 0
                or "disk"
                not in str(
                    (
                        execution.get("cache_detail")
                        if isinstance(execution, dict)
                        else ""
                    )
                    or ""
                ).lower()
            ):
                return set(), []
            if cache_policy == "q4":
                facts.update(
                    {
                        "cross_session_reuse",
                        "disk_refault_observed",
                    }
                )
        all_counter_deltas = [
            row.get("health_counter_deltas")
            for row in requests
            if isinstance(row, dict)
            and isinstance(row.get("health_counter_deltas"), dict)
        ]
        if expected_phase["index"] == 2:
            saw_phase2_ram_eviction = any(
                integer(row.get("scheduler_cache.evictions")) > 0
                for row in all_counter_deltas
            )
            saw_phase2_disk_eviction = any(
                integer(row.get("block_disk_cache.disk_evictions")) > 0
                for row in all_counter_deltas
            )
        phase_instantiated[expected_phase["index"]] = instantiated
        configured = (
            topology.get("configured")
            if isinstance(topology, dict)
            else None
        )
        tq = (
            topology.get("turboquant_kv_cache")
            if isinstance(topology, dict)
            else None
        )
        kv_quant = (
            topology.get("kv_cache_quantization")
            if isinstance(topology, dict)
            else None
        )
        native_cache = (
            topology.get("native_cache")
            if isinstance(topology, dict)
            else None
        )
        if cache_policy == "q4":
            if (
                not isinstance(configured, dict)
                or configured.get("kv_cache_quantization") != "q4"
                or configured.get("kv_cache_quantization_explicit") is not False
                or not isinstance(tq, dict)
                or tq.get("enabled") is not True
                or integer(tq.get("storage_key_bits")) != 4
                or integer(tq.get("storage_value_bits")) != 4
            ):
                return set(), []
            telemetry = tq.get("storage_encode_telemetry")
            if not isinstance(telemetry, dict) or not telemetry:
                return set(), []
            q4_phase_indexes.add(expected_phase["index"])
        elif cache_policy == "ssd-only":
            if (
                representative_id != V5_PRIMARY_REPRESENTATIVE_ID
                or not isinstance(configured, dict)
                or configured.get("kv_cache_quantization") != "none"
                or configured.get("kv_cache_quantization_explicit") is not True
                or not isinstance(tq, dict)
                or tq.get("enabled") is not False
                or not isinstance(kv_quant, dict)
                or kv_quant.get("enabled") is not False
            ):
                return set(), []
            facts.add("explicit_off_honored")
        elif cache_policy == "native":
            derived_native = bundle_snapshot["derived"]["native_cache"]
            generic_tq = (
                native_cache.get("generic_turboquant_kv")
                if isinstance(native_cache, dict)
                else None
            )
            if (
                representative_id != V5_NATIVE_REPRESENTATIVE_ID
                or derived_native
                not in {
                    "minimax_m3_sparse",
                    "dsv4_composite",
                    "openpangu_native",
                    "cca",
                }
                or not isinstance(configured, dict)
                or configured.get("kv_cache_quantization") != "none"
                or configured.get("kv_cache_quantization_explicit") is not False
                or not isinstance(tq, dict)
                or tq.get("enabled") is not False
                or not isinstance(kv_quant, dict)
                or kv_quant.get("enabled") is not False
                or not isinstance(native_cache, dict)
                or not isinstance(generic_tq, dict)
                or generic_tq.get("enabled") is not False
            ):
                return set(), []
            facts.add("unsupported_architecture_exception_cache")
        else:
            return set(), []
        hashes.extend(
            (
                summary_sha256,
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
        )
    try:
        expected_l2_attestation = _v5_derive_l2_size_eviction_attestation(
            run_id=str(artifact.get("run_id") or ""),
            nonce=str(artifact.get("nonce") or ""),
            phase2_summary=phase_summaries[2],
            phase2_summary_sha256=phase_summary_hashes[2],
            phase3_summary=phase_summaries[3],
            phase3_summary_sha256=phase_summary_hashes[3],
        )
    except (KeyError, ValueError):
        return set(), []
    phase2_instantiated = phase_instantiated.get(2)
    phase2_final_disk = nested(
        phase_summaries.get(2) or {},
        "health_final",
        "cache",
        "block_disk_cache",
    )
    phase3_execution = phase_executions.get(3)
    if (
        artifact.get("l2_size_eviction_attestation")
        != expected_l2_attestation
        or not saw_phase2_ram_eviction
        or not saw_phase2_disk_eviction
        or not isinstance(phase2_instantiated, dict)
        or integer(phase2_instantiated.get("block_disk_max_size_bytes"))
        != expected_l2_attestation["saved_max_bytes"]
        or not isinstance(phase2_final_disk, dict)
        or integer(phase2_final_disk.get("disk_size_bytes"))
        != expected_l2_attestation["final_observed_bytes"]
        or not isinstance(phase3_execution, dict)
        or integer(phase3_execution.get("cached_tokens"))
        != expected_l2_attestation["restart_restored_tokens"]
        or integer(phase3_execution.get("disk_blocks"))
        != expected_l2_attestation["restart_disk_blocks"]
        or integer(phase3_execution.get("uncached_prompt_tokens"))
        != expected_l2_attestation["restart_uncached_tokens"]
        or "disk"
        not in str(phase3_execution.get("cache_detail") or "").lower()
    ):
        return set(), []
    facts.update(
        {
            "oldest_unused_evicted",
            "disk_refault_observed",
            "restart_disk_restore",
            "disk_size_limit_enforced",
            "disk_oldest_unused_evicted",
        }
    )
    if (
        paged_modes != {False, True}
        or q4_phase_indexes != {0, 1, 2, 3}
        or policy_roles
        != {
            (V5_PRIMARY_REPRESENTATIVE_ID, "q4"),
            (V5_PRIMARY_REPRESENTATIVE_ID, "ssd-only"),
            (V5_NATIVE_REPRESENTATIVE_ID, "native"),
        }
        or any(len(values) != 1 for values in session_ids.values())
        or session_ids[V5_PRIMARY_REPRESENTATIVE_ID]
        == session_ids[V5_NATIVE_REPRESENTATIVE_ID]
        or len(set(all_backend_pids)) != len(V5_CACHE_PHASES)
        or any(len(set(pids)) != 2 for pids in q4_restart_pids.values())
    ):
        return set(), []
    facts.update(
        {
            "q4_default_when_supported",
            "encode_decode_live",
            representatives[V5_PRIMARY_REPRESENTATIVE_ID]["bundle"]["derived"][
                "native_cache"
            ],
            representatives[V5_NATIVE_REPRESENTATIVE_ID]["bundle"]["derived"][
                "native_cache"
            ],
        }
    )
    return facts, hashes


def _v5_git_snapshot() -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD")
    tree = run_git("rev-parse", "HEAD^{tree}")
    upstream = run_git("rev-parse", "@{upstream}")
    remote_main = run_git("ls-remote", "--exit-code", "origin", "refs/heads/main")
    counts = run_git(
        "rev-list",
        "--left-right",
        "--count",
        "origin/main...HEAD",
    ).split()
    diff_bytes = subprocess.run(
        ["git", "diff", "--name-status", "v1.6.17..HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return {
        "commit": head,
        "tree": tree,
        "status_porcelain": run_git(
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
        "upstream_commit": upstream,
        "remote_main_commit": remote_main.split()[0],
        "main_only": int(counts[0]),
        "branch_only": int(counts[1]),
        "remote_identity": canonical_github_repo(
            run_git("remote", "get-url", "origin")
        ),
        "release_diff_bytes": diff_bytes,
        "release_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def _v5_source_and_scope_facts(
    before: dict[str, Any],
    after: dict[str, Any],
    runtime: dict[str, Any],
    bundle_snapshot: dict[str, Any],
) -> tuple[set[str], set[str]]:
    source_facts: set[str] = set()
    scope_facts: set[str] = set()
    stable_fields = (
        "commit",
        "tree",
        "status_porcelain",
        "upstream_commit",
        "remote_main_commit",
        "main_only",
        "branch_only",
        "remote_identity",
        "release_diff_sha256",
    )
    if any(before.get(key) != after.get(key) for key in stable_fields):
        return source_facts, scope_facts
    head = after.get("commit")
    if head and after.get("tree") and after.get("status_porcelain") == "":
        source_facts.add("checkout_head_exact")
    if head and after.get("upstream_commit") == head:
        source_facts.add("origin_branch_pushed")
    if (
        head
        and after.get("remote_main_commit") == head
        and after.get("main_only") == 0
        and after.get("branch_only") == 0
    ):
        source_facts.add("origin_main_exact")
    if runtime.get("electron") and runtime.get("dom", {}).get("sourceCommit") == head:
        source_facts.update({"electron_revision_exact", "renderer_revision_exact"})
    expected = runtime.get("expected_source_attestation")
    if not isinstance(expected, dict):
        expected = release_runtime_source_attestation()
    runtime_hashes = runtime.get("runtime_hashes")
    if (
        isinstance(runtime_hashes, dict)
        and runtime_hashes.get("package_init_sha256")
        == expected["package_init_sha256"]
        and runtime_hashes.get("server_module_sha256")
        == expected["server_module_sha256"]
        and runtime_hashes.get("python_source_tree_sha256")
        == expected["python_source_tree_sha256"]
    ):
        source_facts.add("python_import_revision_exact")
    if isinstance(runtime.get("backend", {}).get("pid"), int):
        source_facts.add("engine_pid_recorded")
    observed_bundle_fingerprints = {
        str(candidate.get("bundle_fingerprint_sha256") or "")
        for candidate in (
            [runtime]
            + [
                row.get("observation")
                for row in runtime.get("phase_observations", [])
                if isinstance(row, dict)
            ]
        )
        if isinstance(candidate, dict)
    }
    if bundle_snapshot["fingerprint_sha256"] in observed_bundle_fingerprints:
        source_facts.add("bundle_config_hashes_recorded")

    diff_bytes = after.get("release_diff_bytes")
    if isinstance(diff_bytes, bytes) and diff_bytes.strip():
        scope_facts.add("v1_6_17_to_head_diff_reviewed")
        paths = []
        for line in diff_bytes.decode("utf-8", errors="strict").splitlines():
            fields = line.split("\t")
            paths.extend(fields[1:])
        public_forbidden = re.compile(
            r"(?i)(?:^|/)(?:docs/internal|\\.agents|notes|screenshots?|recordings?)(?:/|$)"
        )
        allowed_prefixes = (
            "panel/",
            "vmlx_engine/",
            "tests/",
            "scripts/",
            "pyproject.toml",
            "uv.lock",
            ".gitignore",
            "README",
            "CHANGELOG",
            "LICENSE",
        )
        if paths and all(path.startswith(allowed_prefixes) for path in paths):
            scope_facts.add("all_intended_fixes_mapped")
            scope_facts.add("unintended_changes_none_or_documented")
        if paths and not any(public_forbidden.search(path) for path in paths):
            scope_facts.add("public_repository_hygiene_passed")
    return source_facts, scope_facts


def _v5_run_command(
    check_name: str,
    spec: dict[str, Any],
    run_context: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    argv = [str(value) for value in spec["argv"]]
    executable_pin, executable_invocation = _v5_pin_executable_invocation(
        Path(argv[0]),
        run_dir,
    )
    # Pin every absolute regular-file argument, including executable scripts.
    # npm-cli.js is commonly mode 0755 but is still interpreted by Node; its
    # bytes are therefore part of the owned command identity, not the process
    # executable identity.
    script_pins_by_path = {
        str(Path(value).resolve()): _v5_pin_regular_file(Path(value))
        for value in argv[1:]
        if value.startswith("/") and Path(value).is_file()
    }
    for value in spec.get("tool_files", []):
        value = str(value)
        if not value.startswith("/") or not Path(value).is_file():
            continue
        script_pins_by_path[str(Path(value).resolve())] = _v5_pin_regular_file(
            Path(value),
            allow_readonly_system_hardlink=True,
        )
    script_pins = [
        script_pins_by_path[path] for path in sorted(script_pins_by_path)
    ]
    env = _v5_minimal_env(run_dir, dict(spec.get("env") or {}))
    path_prefix_value = spec.get("path_prefix")
    if path_prefix_value is not None:
        path_prefix = Path(str(path_prefix_value))
        try:
            path_prefix_stat = path_prefix.lstat()
            path_prefix = path_prefix.resolve(strict=True)
        except OSError as exc:
            raise ValueError("owned command PATH prefix is unavailable") from exc
        if (
            not path_prefix.is_absolute()
            or not path_within(path_prefix, run_dir.resolve())
            or stat.S_ISLNK(path_prefix_stat.st_mode)
            or not stat.S_ISDIR(path_prefix_stat.st_mode)
        ):
            raise ValueError("owned command PATH prefix is unsafe")
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    started_at = _iso_now()
    process = subprocess.Popen(  # noqa: S603 - source-owned static plan
        argv,
        cwd=Path(spec["cwd"]).resolve(),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process_observation = _observe_process(process.pid)
    if not _v5_process_matches_pin(process_observation, executable_pin):
        process.kill()
        process.wait()
        raise RuntimeError(f"{check_name} child identity mismatch")
    stdout, stderr = process.communicate()
    ended_at = _iso_now()
    if (
        not _v5_pin_unchanged(executable_pin, executable=True)
        or not _v5_executable_invocation_unchanged(executable_invocation)
        or any(not _v5_pin_unchanged(pin) for pin in script_pins)
    ):
        raise RuntimeError(f"{check_name} executable or script changed")
    return {
        "schema": OWNED_EXECUTION_SCHEMA,
        "run_id": run_context["run_id"],
        "nonce": run_context["nonce"],
        "check": check_name,
        "command_id": spec["command_id"],
        "pid": process.pid,
        "process_observation": process_observation,
        "argv": argv,
        "cwd": str(Path(spec["cwd"]).resolve()),
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": process.returncode,
        "executable": executable_pin,
        "executable_invocation": executable_invocation,
        "scripts": script_pins,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "__stdout_bytes": stdout,
        "__stderr_bytes": stderr,
    }


def _v5_write_executable_wrapper(path: Path, payload: str) -> dict[str, Any]:
    """Create and pin one private, non-symlink executable wrapper."""

    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o700,
    )
    try:
        raw = payload.encode("utf-8")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("could not write fixed tool wrapper")
            view = view[written:]
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or _stable_file_identity(opened) != _stable_file_identity(observed)
    ):
        raise RuntimeError("fixed tool wrapper was replaced")
    return _v5_pin_regular_file(path, executable=True)


def _v5_prepare_node_toolchain(
    run_root: Path,
    *,
    node_path: Path,
    npm_cli_path: Path,
    bin_name: str,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    """Pin Node/npm and expose only private wrappers to nested shebangs."""

    if not re.fullmatch(r"[a-z0-9_.-]+", bin_name):
        raise ValueError("fixed Node toolchain directory name is unsafe")
    node_pin = _v5_pin_regular_file(node_path.resolve(strict=True), executable=True)
    npm_pin = _v5_pin_regular_file(npm_cli_path.resolve(strict=True))
    shell_pin = _v5_pin_regular_file(Path("/bin/sh"), executable=True)
    bin_dir = run_root / bin_name
    bin_dir.mkdir(mode=0o700)
    node_wrapper = _v5_write_executable_wrapper(
        bin_dir / "node",
        "#!/bin/sh\n"
        f"exec {shlex.quote(node_pin['path'])} \"$@\"\n",
    )
    npm_wrapper = _v5_write_executable_wrapper(
        bin_dir / "npm",
        "#!/bin/sh\n"
        f"exec {shlex.quote(node_pin['path'])} "
        f"{shlex.quote(npm_pin['path'])} \"$@\"\n",
    )
    return (
        Path(node_pin["path"]),
        Path(npm_pin["path"]),
        bin_dir,
        [node_pin, npm_pin, shell_pin, node_wrapper, npm_wrapper],
    )


def _v5_hash_output_tree(
    output_root: Path,
    required_relative_paths: tuple[str, ...],
) -> dict[str, Any] | None:
    try:
        root_stat = output_root.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        return None
    files = sorted(path for path in output_root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in output_root.rglob("*")):
        return None
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            pin = _v5_pin_regular_file(path)
        except (OSError, ValueError):
            return None
        rows.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": pin["sha256"],
                "size": pin["size"],
            }
        )
    observed = {row["path"] for row in rows}
    if not set(required_relative_paths) <= observed:
        return None
    return {
        "root": str(output_root),
        "files": rows,
        "tree_sha256": _canonical_json_sha256(rows),
    }


def _v5_parse_vitest_terminal_summary(text: str) -> dict[str, Any] | None:
    """Parse only Vitest's terminal accounting, never arbitrary test logs."""

    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)

    def parse_line(label: str) -> dict[str, Any] | None:
        pattern = re.compile(
            rf"\b{re.escape(label)}\s+(.+?)\s+\((\d+)\)\s*$"
        )
        matches = [
            match
            for line in plain.splitlines()
            if (match := pattern.search(line)) is not None
        ]
        if not matches:
            return None
        match = matches[-1]
        counts: dict[str, int] = {}
        for raw_count, raw_label in re.findall(
            r"\b(\d+)\s+(passed|failed|skipped|todo|errors?)\b",
            match.group(1),
        ):
            name = "error" if raw_label in {"error", "errors"} else raw_label
            counts[name] = counts.get(name, 0) + int(raw_count)
        total = int(match.group(2))
        if not counts or sum(counts.values()) != total:
            return None
        return {"counts": counts, "total": total}

    files = parse_line("Test Files")
    tests = parse_line("Tests")
    if files is None or tests is None:
        return None
    return {"files": files, "tests": tests}


def _v5_owned_check_facts(
    check_name: str,
    executions: list[dict[str, Any]],
    spec: dict[str, Any],
    jang_state: dict[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    if not executions or any(row.get("exit_code") != 0 for row in executions):
        return set(), {}
    combined = b"\n".join(
        row["__stdout_bytes"] + b"\n" + row["__stderr_bytes"]
        for row in executions
    )
    try:
        text = combined.decode("utf-8")
    except UnicodeDecodeError:
        return set(), {}
    if check_name in {"full_python_suite", "full_panel_suite"}:
        if check_name == "full_python_suite":
            plain_text = re.sub(r"\x1b\[[0-9;]*m", "", text)
            terminal_summaries = [
                line
                for line in plain_text.splitlines()
                if re.search(r"\b\d+\s+passed\b", line)
                and re.search(r"\bin\s+[0-9:.]+s?(?:\s|$)", line)
            ]
            if not terminal_summaries:
                return set(), {}
            terminal_summary = terminal_summaries[-1]

            def terminal_count(label: str) -> int:
                match = re.search(
                    rf"\b(\d+)\s+{re.escape(label)}\b",
                    terminal_summary,
                )
                return int(match.group(1)) if match else 0

            passed_count = terminal_count("passed")
            deselected_count = terminal_count("deselected")
            skipped_count = terminal_count("skipped")
            xfailed_count = terminal_count("xfailed")
            xpassed_count = terminal_count("xpassed")
            failed_count = terminal_count("failed")
            error_match = re.search(
                r"\b(\d+)\s+errors?\b",
                terminal_summary,
            )
            error_count = int(error_match.group(1)) if error_match else 0
            baseline = OWNED_SUITE_BASELINES[check_name]
            completed_count = (
                passed_count
                + skipped_count
                + xfailed_count
                + xpassed_count
                + failed_count
                + error_count
            )
            expected_collected = completed_count + deselected_count
            collection_counts = {
                int(value)
                for value in re.findall(
                    r"\bcollected\s+(\d+)\s+items?\b",
                    plain_text,
                )
            }
            valid = bool(
                expected_collected in collection_counts
                and completed_count > 0
                and passed_count >= baseline["passed"]
                and failed_count == 0
                and error_count == 0
            )
            if valid:
                return set(V5_RELEASE_ASSERTIONS[check_name]), {}
            return set(), {}
        else:
            summary = _v5_parse_vitest_terminal_summary(text)
            baseline = OWNED_SUITE_BASELINES[check_name]
            valid = bool(
                summary
                and summary["tests"]["counts"].get("passed", 0)
                >= baseline["passed"]
                and summary["files"]["counts"].get("passed", 0)
                >= baseline["files"]
                and summary["tests"]["counts"].get("failed", 0) == 0
                and summary["tests"]["counts"].get("error", 0) == 0
                and summary["files"]["counts"].get("failed", 0) == 0
                and summary["files"]["counts"].get("error", 0) == 0
            )
        if valid:
            return set(V5_RELEASE_ASSERTIONS[check_name]), {}
        return set(), {}
    if check_name == "typecheck":
        return (
            ({"terminal_summary_passed"}, {})
            if not re.search(r"\berror\s+TS\d+\b", text)
            else (set(), {})
        )
    if check_name == "production_build":
        output = _v5_hash_output_tree(
            Path(spec["output_root"]),
            tuple(spec["required_outputs"]),
        )
        required_log = (
            "bundled JANG provenance matches source",
            "bundled-python: all critical imports ok",
        )
        return (
            (set(V5_RELEASE_ASSERTIONS[check_name]), {"output": output})
            if output is not None and all(value in text for value in required_log)
            else (set(), {})
        )
    if check_name == "jang_runtime_provenance":
        distribution_root = Path(spec["distribution_root"])
        wheels = sorted(distribution_root.glob("*.whl"))
        sdists = sorted(distribution_root.glob("*.tar.gz"))
        manifest = tuple(spec["test_manifest"])
        collected = re.search(r"collected\s+(\d+)\s+items?", text)
        passed = re.search(r"(\d+)\s+passed\b", text)
        import_match = re.search(
            r"^VMLINUX_INSTALLED_IMPORT=(\{.*\})$",
            text,
            re.MULTILINE,
        )
        test_import_match = re.search(
            r"^VMLINUX_TEST_IMPORT=(\{.*\})$",
            text,
            re.MULTILINE,
        )
        if (
            len(wheels) != 1
            or len(sdists) != 1
            or collected is None
            or passed is None
            or int(collected.group(1)) != int(passed.group(1))
            or int(passed.group(1)) < int(spec["minimum_test_count"])
            or {row.get("command_id") for row in executions}
            != {"jang_build", "jang_venv", "jang_install", "jang_import", "jang_test"}
            or import_match is None
            or test_import_match is None
        ):
            return set(), {}
        imported = json.loads(import_match.group(1))
        test_imported = json.loads(test_import_match.group(1))
        installed_root = Path(spec["isolated_venv"]).resolve()
        imported_path = Path(str(imported.get("file") or "")).resolve()
        test_imported_path = Path(str(test_imported.get("file") or "")).resolve()
        if (
            not path_within(imported_path, installed_root)
            or not path_within(test_imported_path, installed_root)
            or test_imported_path != imported_path
            or imported.get("version") != JANG_VERSION
            or test_imported.get("version") != JANG_VERSION
            or test_imported.get("source_manifest_sha256")
            != test_imported.get("installed_manifest_sha256")
            or not isinstance(test_imported.get("package_file_count"), int)
            or test_imported["package_file_count"] <= 0
            or test_imported.get("laguna_mixed_affine_shape_bits") != [6, 6]
            or jang_state.get("version") != JANG_VERSION
            or jang_state.get("commit") != JANG_COMMIT
            or jang_state.get("tree") != JANG_TREE
        ):
            return set(), {}
        artifacts = [
            _v5_pin_regular_file(wheels[0]),
            _v5_pin_regular_file(sdists[0]),
            _v5_pin_regular_file(imported_path),
        ]
        return set(V5_RELEASE_ASSERTIONS[check_name]), {
            "artifacts": artifacts,
            "test_manifest": list(manifest),
            "observed_test_count": int(passed.group(1)),
        }
    return set(), {}


def _v5_default_owned_check_plans(
    run_dir: Path,
    jang_root: Path,
) -> dict[str, dict[str, Any]]:
    python = ROOT / ".venv/bin/python"
    node_value = shutil.which("node")
    npm_value = shutil.which("npm")
    uv_value = shutil.which("uv")
    if not python.is_file() or not node_value or not npm_value or not uv_value:
        raise RuntimeError(
            "release Python, Node, npm, or uv executable is unavailable"
        )
    uv = Path(uv_value).resolve(strict=True)
    node, npm_cli, fixed_bin, toolchain_pins = _v5_prepare_node_toolchain(
        run_dir,
        node_path=Path(node_value),
        npm_cli_path=Path(npm_value),
        bin_name="owned-node-bin",
    )
    release_verifier_pins = {
        "NODE": _v5_pin_regular_file(node, executable=True),
        **{
            name: _v5_pin_regular_file(
                path,
                executable=True,
                allow_readonly_system_hardlink=True,
            )
            for name, path in {
                "GIT": Path("/usr/bin/git"),
                "SHASUM": Path("/usr/bin/shasum"),
                "AWK": Path("/usr/bin/awk"),
                "FIND": Path("/usr/bin/find"),
            }.items()
        },
    }
    toolchain_files = sorted(
        {
            *(pin["path"] for pin in toolchain_pins),
            *(pin["path"] for pin in release_verifier_pins.values()),
        }
    )
    production_env = {
        "VMLX_RELEASE_SCOPE": SCOPE,
        "VMLX_JANG_TOOLS_SOURCE": str(jang_root.resolve()),
        "VMLX_BUNDLE_MLX_PLATFORM": "compat",
        "VMLX_EXPECTED_MLX_WHEEL_PLATFORM": "macosx_14_0_arm64",
    }
    for name, pin in release_verifier_pins.items():
        production_env[f"VMLX_R18_TOOL_{name}_REALPATH"] = pin["path"]
        production_env[f"VMLX_R18_TOOL_{name}_SHA256"] = pin["sha256"]
    production_tool_files = sorted(
        {
            *toolchain_files,
            str((ROOT / "panel/package.json").resolve(strict=True)),
            str((ROOT / "panel/scripts/bundle-python.sh").resolve(strict=True)),
            str(
                (ROOT / "panel/scripts/verify-bundled-python.sh").resolve(
                    strict=True
                )
            ),
            str((ROOT / "panel/electron.vite.config.ts").resolve(strict=True)),
            str(
                (
                    ROOT
                    / "panel/node_modules/electron-vite/bin/electron-vite.js"
                ).resolve(strict=True)
            ),
        }
    )
    build_root = run_dir / "electron-build"
    build_root.mkdir(mode=0o700)
    jang_dist = run_dir / "jang-dist"
    jang_dist.mkdir(mode=0o700)
    jang_venv = run_dir / "jang-installed"
    isolated_site_packages = (
        jang_venv
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    test_manifest = (
        str((ROOT / "tests/test_laguna_loader.py").resolve(strict=True)),
        str((ROOT / "tests/test_jang_affine_storage.py").resolve(strict=True)),
        str((ROOT / "tests/test_jang_loader.py").resolve(strict=True)),
        str(jang_root.resolve() / "tests/test_laguna_jang_affine_policy.py"),
    )
    tracked_jang_files = tuple(
        line
        for line in run_git_in(jang_root, "ls-files", "jang_tools").splitlines()
        if line.startswith("jang_tools/")
    )
    if not tracked_jang_files:
        raise RuntimeError("clean JANG source has no tracked package files")
    jang_source_files = [
        str((jang_root / relative).resolve(strict=True))
        for relative in tracked_jang_files
    ]
    jang_test_files = [
        *test_manifest,
        str((ROOT / "pytest.ini").resolve(strict=True)),
        str((jang_root / "pyproject.toml").resolve(strict=True)),
        *jang_source_files,
    ]
    import_script = (
        "import importlib.metadata,json,sys,time;"
        f"sys.path.insert(0,{str(isolated_site_packages)!r});"
        "import jang_tools;"
        "print('VMLINUX_INSTALLED_IMPORT='+json.dumps({"
        "'file':jang_tools.__file__,"
        "'version':importlib.metadata.version('jang')"
        "},sort_keys=True),flush=True);"
        "time.sleep(0.2)"
    )
    test_import_script = (
        "import hashlib,importlib.metadata,json,pathlib,runpy,sys,zipfile;"
        "wheel_path=pathlib.Path(sys.argv[1]).resolve(strict=True);"
        f"sys.path.insert(0,{str(isolated_site_packages)!r});"
        "import jang_tools;"
        "from jang_tools.laguna.runtime import infer_affine_bits_from_shapes;"
        "shape_results=["
        "infer_affine_bits_from_shapes((1,576),(1,48),group_size=64,fallback_bits=8),"
        "infer_affine_bits_from_shapes((1,384),(1,32),group_size=64,fallback_bits=8)];"
        "assert shape_results==[6,6],shape_results;"
        f"source_root=pathlib.Path({str((jang_root / 'jang_tools').resolve())!r});"
        "installed_root=pathlib.Path(jang_tools.__file__).resolve().parent;"
        "archive=zipfile.ZipFile(wheel_path);"
        "wheel_files={name.removeprefix('jang_tools/'):archive.read(name)"
        " for name in archive.namelist()"
        " if name.startswith('jang_tools/') and not name.endswith('/')};"
        "archive.close();"
        "assert wheel_files;"
        "rows=[];missing=[];mismatched=[];"
        "installed_files={path.relative_to(installed_root).as_posix()"
        " for path in installed_root.rglob('*') if path.is_file()"
        " and '__pycache__' not in path.parts and path.suffix!='.pyc'};"
        "assert installed_files==set(wheel_files),(installed_files^set(wheel_files));"
        "[(missing.append(rel) if not (source_root/rel).is_file()"
        " or not (installed_root/rel).is_file() else"
        " (mismatched.append(rel) if (source_root/rel).read_bytes()!=data"
        " or (installed_root/rel).read_bytes()!=data else"
        " rows.append([rel,hashlib.sha256(data).hexdigest()])))"
        " for rel,data in sorted(wheel_files.items())];"
        "assert not missing,missing;assert not mismatched,mismatched;"
        "source_manifest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest();"
        "installed_manifest=source_manifest;"
        "print('VMLINUX_TEST_IMPORT='+json.dumps({"
        "'file':jang_tools.__file__,"
        "'version':importlib.metadata.version('jang'),"
        "'source_manifest_sha256':source_manifest,"
        "'installed_manifest_sha256':installed_manifest,"
        "'package_file_count':len(rows),"
        "'laguna_mixed_affine_shape_bits':shape_results"
        "},sort_keys=True),flush=True);"
        f"sys.path.append({str(ROOT.resolve())!r});"
        f"sys.argv={['pytest', '-s', '-p', 'no:cacheprovider', '--import-mode=importlib', '--rootdir', str(ROOT.resolve()), '-c', str((ROOT / 'pytest.ini').resolve(strict=True)), *test_manifest]!r};"
        "runpy.run_module('pytest',run_name='__main__')"
    )
    return {
        "full_python_suite": {
            "commands": [
                {
                    "command_id": "full_python_suite",
                    "argv": [
                        str(python),
                        "-m",
                        "pytest",
                        "-s",
                        "-p",
                        "no:cacheprovider",
                    ],
                    "cwd": ROOT,
                    "env": {},
                }
            ]
        },
        "full_panel_suite": {
            "commands": [
                {
                    "command_id": "full_panel_suite",
                    "argv": [str(node), str(npm_cli), "test"],
                    "cwd": ROOT / "panel",
                    "env": {},
                    "path_prefix": str(fixed_bin),
                    "tool_files": toolchain_files,
                }
            ]
        },
        "typecheck": {
            "commands": [
                {
                    "command_id": "typecheck",
                    "argv": [str(node), str(npm_cli), "run", "typecheck"],
                    "cwd": ROOT / "panel",
                    "env": {},
                    "path_prefix": str(fixed_bin),
                    "tool_files": toolchain_files,
                }
            ]
        },
        "production_build": {
            "commands": [
                {
                    "command_id": "production_build",
                    "argv": [
                        str(node),
                        str(npm_cli),
                        "run",
                        "build",
                        "--",
                        "--outDir",
                        str(build_root),
                    ],
                    "cwd": ROOT / "panel",
                    "env": production_env,
                    "path_prefix": str(fixed_bin),
                    "tool_files": production_tool_files,
                }
            ],
            "output_root": str(build_root),
            "required_outputs": (
                "main/index.mjs",
                "preload/index.js",
                "renderer/index.html",
            ),
        },
        "jang_runtime_provenance": {
            "commands": [
                {
                    "command_id": "jang_build",
                    "argv": [
                        str(uv),
                        "build",
                        "--python",
                        str(python.resolve(strict=True)),
                        "--no-python-downloads",
                        "--out-dir",
                        str(jang_dist),
                        str(jang_root),
                    ],
                    "cwd": run_dir,
                    "env": {},
                    "tool_files": [
                        str((jang_root / "pyproject.toml").resolve(strict=True)),
                        *jang_source_files,
                    ],
                },
                {
                    "command_id": "jang_venv",
                    "argv": [
                        str(python),
                        "-m",
                        "venv",
                        str(jang_venv),
                    ],
                    "cwd": ROOT,
                    "env": {},
                },
                {
                    "command_id": "jang_install",
                    "argv": [
                        str(jang_venv / "bin/python"),
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-index",
                        str(jang_dist / "*.whl"),
                    ],
                    "cwd": ROOT,
                    "env": {},
                    "expand_single_glob": True,
                },
                {
                    "command_id": "jang_import",
                    "argv": [
                        str(python),
                        "-I",
                        "-c",
                        import_script,
                    ],
                    "cwd": run_dir,
                    "env": {},
                    "tool_files": jang_source_files,
                },
                {
                    "command_id": "jang_test",
                    "argv": [
                        str(python),
                        "-I",
                        "-c",
                        test_import_script,
                        str(jang_dist / "*.whl"),
                    ],
                    "cwd": ROOT,
                    "env": {},
                    "tool_files": jang_test_files,
                    "expand_single_glob": True,
                },
            ],
            "distribution_root": str(jang_dist),
            "isolated_venv": str(jang_venv),
            "test_manifest": test_manifest,
            "minimum_test_count": len(test_manifest),
        },
    }


def _v5_expand_owned_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("expand_single_glob") is not True:
        return spec
    argv = list(spec["argv"])
    expanded: list[str] = []
    for value in argv:
        if "*" not in value or not Path(value).is_absolute():
            expanded.append(value)
            continue
        matches = sorted(Path(value).parent.glob(Path(value).name))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one artifact for {value}")
        expanded.append(str(matches[0]))
    result = dict(spec)
    result["argv"] = expanded
    return result


def _v5_execute_owned_release_checks(
    plans: dict[str, dict[str, Any]],
    run_context: dict[str, Any],
    run_dir: Path,
    jang_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for check_name in OWNED_CHECK_NAMES:
        plan = plans.get(check_name)
        if not isinstance(plan, dict):
            results[check_name] = {"executions": [], "facts": set(), "details": {}}
            continue
        executions = [
            _v5_run_command(
                check_name,
                _v5_expand_owned_spec(command),
                run_context,
                run_dir,
            )
            for command in plan.get("commands") or []
        ]
        facts, details = _v5_owned_check_facts(
            check_name,
            executions,
            plan,
            jang_state,
        )
        results[check_name] = {
            "executions": executions,
            "facts": facts,
            "details": details,
        }
    return results


def _v5_default_producer_plans(
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    python = ROOT / ".venv/bin/python"
    common = [
        str(python),
        str(Path(__file__).resolve()),
        "--v5-owned-worker",
        "{PRODUCER}",
        "--v5-output-fd",
        "{OUTPUT_FD}",
        "--v5-run-id",
        "{RUN_ID}",
        "--v5-nonce",
        "{NONCE}",
        "--v5-session-binding-path",
        "{SESSION_BINDING_PATH}",
        "--v5-ready-path",
        "{READY_PATH}",
        "--v5-release-path",
        "{RELEASE_PATH}",
        "--v5-phase-control-dir",
        "{PHASE_CONTROL_DIR}",
        "--v5-paired-api-path",
        "{PAIRED_API_PATH}",
        "--v5-cache-artifact-root",
        "{CACHE_ARTIFACT_ROOT}",
        "--v5-source-commit",
        "{SOURCE_COMMIT}",
        "--v5-source-tree",
        "{SOURCE_TREE}",
        "--v5-run-intent-path",
        "{RUN_INTENT_PATH}",
        "--v5-run-intent-sha256",
        "{RUN_INTENT_SHA256}",
        "--v5-active-phase-index",
        "{ACTIVE_PHASE_INDEX}",
        "--v5-ui-session-attestation-path",
        "{UI_SESSION_ATTESTATION_PATH}",
        "--v5-previous-backend-pid",
        "{PREVIOUS_BACKEND_PID}",
        "--v5-reuse-session-id",
        "{REUSE_SESSION_ID}",
        "--v5-reuse-session-attestation-path",
        "{REUSE_SESSION_ATTESTATION_PATH}",
        "--bundle-root",
        str(args.bundle_root),
        "--native-bundle-root",
        str(args.native_bundle_root),
        "--direct-base-url",
        args.direct_base_url,
        "--gateway-base-url",
        args.gateway_base_url,
        "--health-url",
        args.health_url,
        "--gateway-health-url",
        args.gateway_health_url,
        "--cdp-url",
        args.cdp_url,
        "--electron-pid",
        str(args.electron_pid),
        "--gateway-pid",
        str(args.gateway_pid),
        "--model",
        args.model,
        "--native-model",
        args.native_model,
    ]
    return {
        producer: {
            "argv": [
                producer if value == "{PRODUCER}" else value for value in common
            ],
            "cwd": ROOT,
            "env": {},
            "ready_timeout_seconds": 1_800,
            "producer_timeout_seconds": 7_200,
        }
        for producer in V5_PRODUCER_NAMES
    }


def _v5_validate_session_binding(
    binding: dict[str, Any],
    ready: dict[str, Any],
    binding_bytes: bytes,
    ui_handle: dict[str, Any],
    run_context: dict[str, Any],
    expected: dict[str, Any],
    phase: dict[str, Any],
) -> str:
    digest = hashlib.sha256(binding_bytes).hexdigest()
    process = ui_handle["process"]
    required_binding = {
        "schema": V5_SESSION_BINDING_SCHEMA,
        "run_id": run_context["run_id"],
        "nonce": run_context["nonce"],
        "ui_producer_pid": process.pid,
        "source_commit": expected["source_commit"],
        "source_tree": expected["source_tree"],
        "model": expected["model"],
        "model_bundle_path": expected["model_bundle_path"],
        "bundle_fingerprint_sha256": expected["bundle_fingerprint_sha256"],
        "direct_base_url": expected["direct_base_url"],
        "gateway_base_url": expected["gateway_base_url"],
        "health_url": expected["health_url"],
        "direct_health_url": expected["direct_health_url"],
        "gateway_health_url": expected["gateway_health_url"],
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "kv_cache_quantization": phase["kv_cache_quantization"],
        "tq_policy": phase["tq_policy"],
        "session_policy": phase["session_policy"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "paged_ram": phase["paged_ram"],
    }
    if any(binding.get(key) != value for key, value in required_binding.items()):
        raise RuntimeError("UI session binding does not match the requested run")
    required_ready = {
        "schema": V5_UI_READY_SCHEMA,
        "run_id": run_context["run_id"],
        "nonce": run_context["nonce"],
        "ui_producer_pid": process.pid,
        "session_id": binding.get("session_id"),
        "binding_sha256": digest,
        "held": True,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "kv_cache_quantization": phase["kv_cache_quantization"],
        "tq_policy": phase["tq_policy"],
        "session_policy": phase["session_policy"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "paged_ram": phase["paged_ram"],
    }
    if any(ready.get(key) != value for key, value in required_ready.items()):
        raise RuntimeError("UI ready sentinel does not match its session binding")
    if not str(binding.get("session_id") or "").strip():
        raise RuntimeError("UI session binding has no session ID")
    for field in ("backend_pid", "gateway_pid", "electron_pid"):
        if not isinstance(binding.get(field), int) or int(binding[field]) <= 0:
            raise RuntimeError(f"UI session binding has invalid {field}")
    start_ordinal = binding.get("session_start_ordinal")
    if not isinstance(start_ordinal, int) or start_ordinal != phase["index"] + 1:
        raise RuntimeError("UI session binding has invalid restart ordinal")
    previous_backend_pid = binding.get("previous_backend_pid")
    if phase["restart_required"]:
        if (
            not isinstance(previous_backend_pid, int)
            or previous_backend_pid <= 0
            or previous_backend_pid == binding["backend_pid"]
        ):
            raise RuntimeError("UI phase did not attest a real backend restart")
    elif previous_backend_pid is not None:
        raise RuntimeError("initial UI phase unexpectedly names an old backend")
    for field in ("cdp_url", "direct_base_url", "gateway_base_url", "health_url"):
        parsed = urlsplit(str(binding.get(field) or ""))
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RuntimeError(f"UI session binding has unsafe {field}")
    ready_at = _parse_aware_timestamp(ready.get("ready_at"))
    now = datetime.now(UTC)
    if (
        ready_at is None
        or ready_at > now
        or (now - ready_at).total_seconds() > 7_200
    ):
        raise RuntimeError("UI ready sentinel has no aware timestamp")
    return digest


def _v5_default_hold_observation(binding: dict[str, Any]) -> dict[str, Any]:
    health_bytes = _v5_loopback_http_get(str(binding["health_url"]))
    dom_bytes = _v5_cdp_dom_snapshot(str(binding["cdp_url"]))
    try:
        health = json.loads(health_bytes)
        dom = json.loads(dom_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("held UI runtime returned malformed observation") from exc
    if not isinstance(health, dict) or not isinstance(dom, dict):
        raise RuntimeError("held UI runtime observation is not object JSON")
    runtime = health.get("runtime_provenance")
    bundle_snapshot = _read_bundle_directory_snapshot(
        binding.get("model_bundle_path")
    )
    if not isinstance(runtime, dict) or bundle_snapshot is None:
        raise RuntimeError("held UI runtime source/model/session mismatch")
    backend = _observe_process(int(binding["backend_pid"]))
    gateway = _observe_process(int(binding["gateway_pid"]))
    electron = _observe_process(int(binding["electron_pid"]))
    if not all(isinstance(row, dict) for row in (backend, gateway, electron)):
        raise RuntimeError("held UI runtime process observation is incomplete")
    direct_listener = _observe_listener(
        *_v5_host_port(str(binding["direct_base_url"]))
    )
    gateway_listener = _observe_listener(
        *_v5_host_port(str(binding["gateway_base_url"]))
    )
    if (
        direct_listener.get("owner_pid") != backend["pid"]
        or gateway_listener.get("owner_pid") != gateway["pid"]
    ):
        raise RuntimeError("held UI runtime listener ownership mismatch")
    sessions = dom.get("session_ids")
    if (
        bundle_snapshot.get("fingerprint_sha256")
        != binding["bundle_fingerprint_sha256"]
        or not _v5_validate_runtime_bundle_attestation(
            health,
            runtime,
            bundle_snapshot,
        )
        or dom.get("sourceCommit") != binding["source_commit"]
        or not isinstance(sessions, list)
        or binding["session_id"] not in sessions
    ):
        raise RuntimeError("held UI runtime source/model/session mismatch")
    return {
        "observed_at": _iso_now(),
        "health_bytes_sha256": hashlib.sha256(health_bytes).hexdigest(),
        "dom_bytes_sha256": hashlib.sha256(dom_bytes).hexdigest(),
        "backend": backend,
        "gateway": gateway,
        "electron": electron,
        "direct_listener": direct_listener,
        "gateway_listener": gateway_listener,
    }


def _v5_wait_for_ui_hold(
    ui_handle: dict[str, Any],
    ready_path: Path,
    binding_path: Path,
    run_context: dict[str, Any],
    expected: dict[str, Any],
    phase: dict[str, Any],
    *,
    timeout: float,
    observer: Any,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    process = ui_handle["process"]
    last_read_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("UI producer exited before declaring a live hold")
        if ready_path.exists() or ready_path.is_symlink():
            if not (binding_path.exists() or binding_path.is_symlink()):
                raise RuntimeError("UI ready sentinel appeared before its binding")
            try:
                binding, binding_bytes = _v5_read_owned_json(binding_path)
                ready, _ = _v5_read_owned_json(ready_path)
                binding_digest = _v5_validate_session_binding(
                    binding,
                    ready,
                    binding_bytes,
                    ui_handle,
                    run_context,
                    expected,
                    phase,
                )
            except (OSError, RuntimeError) as exc:
                last_read_error = exc
                time.sleep(0.05)
                continue
            try:
                observation = observer(binding)
            except RuntimeError as exc:
                # Clearing the renderer's minimum-width emulation is
                # asynchronous at the CDP boundary. The source-owned UI child
                # can publish its immutable binding just before Chromium has
                # reported the restored layout viewport. Retry only that exact
                # transient; every provenance or ownership mismatch still
                # fails immediately.
                if str(exc) != "CDP viewport did not restore after clearing override":
                    raise
                last_read_error = exc
                time.sleep(0.05)
                continue
            if not isinstance(observation, dict):
                raise RuntimeError("held UI observer returned no raw observation")
            return binding, binding_digest, observation
        if binding_path.exists() or binding_path.is_symlink():
            # The binding can be written just before the ready sentinel, but it
            # must not sit unpaired long enough to be mistaken for a stale run.
            last_read_error = RuntimeError("UI binding is not yet ready-bound")
        time.sleep(0.05)
    if last_read_error is not None:
        raise RuntimeError("UI hold did not become valid before timeout") from (
            last_read_error
        )
    raise RuntimeError("UI producer timed out before declaring a live hold")


def _v5_require_child_hold_binding(
    result: dict[str, Any],
    *,
    session_id: str,
    binding_sha256: str,
    phase: dict[str, Any] | None = None,
) -> None:
    envelope = result.get("capture")
    if (
        not isinstance(envelope, dict)
        or envelope.get("session_id") != session_id
        or envelope.get("session_binding_sha256") != binding_sha256
        or envelope.get("captured_during_ui_hold") is not True
    ):
        raise RuntimeError(
            f"{result.get('name')} producer is not bound to the held UI session"
        )
    if phase is not None and any(
        envelope.get(key) != value
        for key, value in {
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
        }.items()
    ):
        raise RuntimeError(
            f"{result.get('name')} producer is not bound to its active phase"
        )


def _v5_wait_for_cache_phase_done(
    cache_handle: dict[str, Any],
    done_path: Path,
    *,
    run_context: dict[str, Any],
    phase: dict[str, Any],
    binding: dict[str, Any],
    binding_sha256: str,
    timeout: float,
) -> tuple[dict[str, Any], bytes]:
    deadline = time.monotonic() + timeout
    process = cache_handle["process"]
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"cache producer exited before phase {phase['name']} completed"
            )
        if done_path.exists() or done_path.is_symlink():
            try:
                done, done_bytes = _v5_read_owned_json(done_path)
                expected = {
                    "schema": V5_CACHE_PHASE_DONE_SCHEMA,
                    "run_id": run_context["run_id"],
                    "nonce": run_context["nonce"],
                    "phase_index": phase["index"],
                    "phase_name": phase["name"],
                    "representative_id": phase["representative_id"],
                    "bundle_role": phase["bundle_role"],
                    "cache_policy": phase["cache_policy"],
                    "kv_cache_quantization": phase["kv_cache_quantization"],
                    "tq_policy": phase["tq_policy"],
                    "session_policy": phase["session_policy"],
                    "operation": phase["operation"],
                    "ui_action_profile": phase["ui_action_profile"],
                    "ui_turn_count": phase["ui_turn_count"],
                    "api_action_profile": phase["api_action_profile"],
                    "paged_ram": phase["paged_ram"],
                    "session_id": binding["session_id"],
                    "backend_pid": binding["backend_pid"],
                    "session_binding_sha256": binding_sha256,
                }
                if any(done.get(key) != value for key, value in expected.items()):
                    raise RuntimeError("cache phase completion binding mismatch")
                summary_sha256 = str(done.get("summary_sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", summary_sha256):
                    raise RuntimeError("cache phase completion lacks summary hash")
                completed_at = _parse_aware_timestamp(done.get("completed_at"))
                now = datetime.now(UTC)
                if (
                    completed_at is None
                    or completed_at > now
                    or (now - completed_at).total_seconds() > 7_200
                ):
                    raise RuntimeError("cache phase completion is stale")
                return done, done_bytes
            except (OSError, RuntimeError) as exc:
                last_error = exc
        time.sleep(0.05)
    if last_error is not None:
        raise RuntimeError(
            f"cache phase {phase['name']} did not become valid"
        ) from last_error
    raise RuntimeError(f"cache phase {phase['name']} timed out")


def _v5_wait_for_ui_session_attestation(
    ui_handle: dict[str, Any],
    path: Path,
    *,
    run_context: dict[str, Any],
    run_intent_sha256: str,
    phase: dict[str, Any],
    binding: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], bytes]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if ui_handle["process"].poll() is not None:
            raise RuntimeError(
                "UI producer exited before writing its session attestation"
            )
        if path.exists() or path.is_symlink():
            try:
                attestation, attestation_bytes = _v5_read_owned_json(path)
                _v5_validate_ui_session_attestation(
                    attestation,
                    run_context=run_context,
                    run_intent_sha256=run_intent_sha256,
                    phase=phase,
                    binding=binding,
                )
                return attestation, attestation_bytes
            except (OSError, RuntimeError) as exc:
                last_error = exc
        time.sleep(0.05)
    if last_error is not None:
        raise RuntimeError(
            "UI session attestation did not become valid"
        ) from last_error
    raise RuntimeError("UI session attestation timed out")


def _v5_validate_ui_session_attestation(
    attestation: dict[str, Any],
    *,
    run_context: dict[str, Any],
    run_intent_sha256: str,
    phase: dict[str, Any],
    binding: dict[str, Any],
) -> None:
    expected_fields = {
        "schema",
        "run_id",
        "nonce",
        "run_intent_sha256",
        "phase_index",
        "phase_name",
        "representative_id",
        "bundle_role",
        "cache_policy",
        "paged_ram",
        "ui_action_profile",
        "ui_turn_count",
        "api_action_profile",
        "ui_producer_pid",
        "session_id",
        "model",
        "model_bundle_path",
        "bundle_fingerprint_sha256",
        "backend_pid",
        "gateway_pid",
        "direct_base_url",
        "gateway_base_url",
        "electron_pid",
        "cdp_origin",
        "lifecycle_owner",
        "source_commit",
        "source_tree",
        "renderer_source_sha256",
        "session_binding_sha256",
        "created_at",
    }
    expected_values = {
        "schema": V5_UI_SESSION_ATTESTATION_SCHEMA,
        "run_id": run_context["run_id"],
        "nonce": run_context["nonce"],
        "run_intent_sha256": run_intent_sha256,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "paged_ram": phase["paged_ram"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "ui_producer_pid": binding["ui_producer_pid"],
        "session_id": binding["session_id"],
        "model": binding["model"],
        "model_bundle_path": binding["model_bundle_path"],
        "bundle_fingerprint_sha256": binding[
            "bundle_fingerprint_sha256"
        ],
        "backend_pid": binding["backend_pid"],
        "gateway_pid": binding["gateway_pid"],
        "direct_base_url": binding["direct_base_url"],
        "gateway_base_url": binding["gateway_base_url"],
        "electron_pid": binding["electron_pid"],
        "cdp_origin": binding["cdp_url"],
        "lifecycle_owner": "parent",
        "source_commit": binding["source_commit"],
        "source_tree": binding["source_tree"],
        "session_binding_sha256": binding["harness_binding_sha256"],
    }
    if set(attestation) != expected_fields or any(
        attestation.get(key) != value
        for key, value in expected_values.items()
    ):
        raise RuntimeError("UI session attestation binding mismatch")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(attestation.get("renderer_source_sha256") or ""),
    ):
        raise RuntimeError("UI session attestation renderer hash is invalid")
    created_at = _parse_aware_timestamp(attestation.get("created_at"))
    now = datetime.now(UTC)
    if (
        created_at is None
        or created_at > now
        or (now - created_at).total_seconds() > 7_200
    ):
        raise RuntimeError("UI session attestation timestamp is invalid")


def _v5_require_multiphase_cache_binding(
    result: dict[str, Any],
    phase_bindings: list[dict[str, Any]],
) -> None:
    envelope = result.get("capture")
    expected = [
        {
            "phase_index": item["phase"]["index"],
            "phase_name": item["phase"]["name"],
            "representative_id": item["phase"]["representative_id"],
            "cache_policy": item["phase"]["cache_policy"],
            "kv_cache_quantization": item["phase"][
                "kv_cache_quantization"
            ],
            "tq_policy": item["phase"]["tq_policy"],
            "session_policy": item["phase"]["session_policy"],
            "ui_action_profile": item["phase"]["ui_action_profile"],
            "ui_turn_count": item["phase"]["ui_turn_count"],
            "api_action_profile": item["phase"]["api_action_profile"],
            "session_id": item["binding"]["session_id"],
            "model": item["binding"]["model"],
            "bundle_fingerprint_sha256": item["binding"][
                "bundle_fingerprint_sha256"
            ],
            "backend_pid": item["binding"]["backend_pid"],
            "session_binding_sha256": item["binding_sha256"],
        }
        for item in phase_bindings
    ]
    if (
        not isinstance(envelope, dict)
        or envelope.get("captured_during_ui_hold") is not True
        or envelope.get("phase_bindings") != expected
    ):
        raise RuntimeError("cache producer is not bound to every held UI phase")


def _v5_run_intent_phase_plan(
    expected_binding: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for phase in V5_CACHE_PHASES:
        representative = expected_binding.get(phase["representative_id"])
        if not isinstance(representative, dict):
            raise ValueError("run intent representative set is incomplete")
        native_cache_policy = (
            "generic-kv-tq"
            if phase["representative_id"] == V5_PRIMARY_REPRESENTATIVE_ID
            else representative.get("native_cache_policy")
        )
        if (
            not isinstance(native_cache_policy, str)
            or not native_cache_policy
        ):
            raise ValueError("run intent native-cache policy is missing")
        plan.append(
            {
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": phase["representative_id"],
                "bundle_role": phase["bundle_role"],
                "cache_policy": phase["cache_policy"],
                "kv_cache_quantization": phase["kv_cache_quantization"],
                "tq_policy": phase["tq_policy"],
                "native_cache_policy": native_cache_policy,
                "session_policy": phase["session_policy"],
                "paged_ram": phase["paged_ram"],
                "operation": phase["operation"],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "restart_required": phase["restart_required"],
                "model": representative["model"],
                "model_bundle_path": representative["model_bundle_path"],
                "bundle_fingerprint_sha256": representative[
                    "bundle_fingerprint_sha256"
                ],
            }
        )
    return plan


def _v5_build_run_intent(
    run_context: dict[str, Any],
    expected_binding: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(expected_binding) != set(V5_REPRESENTATIVE_IDS):
        raise ValueError("run intent representative set is incomplete")
    common_values = {
        (
            row.get("source_commit"),
            row.get("source_tree"),
            row.get("direct_base_url"),
            row.get("gateway_base_url"),
            row.get("direct_health_url"),
            row.get("gateway_health_url"),
        )
        for row in expected_binding.values()
        if isinstance(row, dict)
    }
    if len(common_values) != 1:
        raise ValueError("run intent representatives do not share one source")
    (
        source_commit,
        source_tree,
        direct_base_url,
        gateway_base_url,
        direct_health_url,
        gateway_health_url,
    ) = next(iter(common_values))
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(source_commit or ""))
        or not re.fullmatch(r"[0-9a-f]{40}", str(source_tree or ""))
    ):
        raise ValueError("run intent source identity is invalid")
    created_at = str(run_context.get("created_at") or "")
    if _parse_aware_timestamp(created_at) is None:
        raise ValueError("run intent created_at is not an aware timestamp")
    harnesses = {
        name: {
            "relative_path": relative_path,
            "sha256": _v5_pin_regular_file(ROOT / relative_path)["sha256"],
        }
        for name, relative_path in V5_RUN_INTENT_HARNESSES.items()
    }
    payload: dict[str, Any] = {
        "schema": V5_RUN_INTENT_SCHEMA,
        "run_id": run_context["run_id"],
        "nonce": run_context["nonce"],
        "created_at": created_at,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "harnesses": harnesses,
        "direct_base_url": direct_base_url,
        "gateway_base_url": gateway_base_url,
        "direct_health_url": direct_health_url,
        "gateway_health_url": gateway_health_url,
        "l2_size_eviction_requirements": dict(
            V5_L2_SIZE_EVICTION_REQUIREMENTS
        ),
        "phase_plan": _v5_run_intent_phase_plan(expected_binding),
    }
    payload["canonical_sha256"] = _canonical_json_sha256(payload)
    return payload


def _v5_validate_run_intent(
    intent: dict[str, Any],
    run_context: dict[str, Any],
    expected_binding: dict[str, dict[str, Any]],
) -> None:
    expected = _v5_build_run_intent(run_context, expected_binding)
    if intent != expected:
        raise RuntimeError("owned run intent does not match the canonical plan")
    unsigned = dict(intent)
    canonical_sha256 = unsigned.pop("canonical_sha256", None)
    if canonical_sha256 != _canonical_json_sha256(unsigned):
        raise RuntimeError("owned run intent canonical hash is invalid")


def _v5_execute_producers(
    plans: dict[str, dict[str, Any]],
    run_context: dict[str, Any],
    run_dir: Path,
    *,
    expected_binding: dict[str, dict[str, Any]],
    hold_observer: Any = _v5_default_hold_observation,
) -> dict[str, dict[str, Any]]:
    if set(plans) != set(V5_PRODUCER_NAMES):
        raise ValueError("owned producer plan set is incomplete")
    if set(expected_binding) != set(V5_REPRESENTATIVE_IDS):
        raise ValueError("owned representative binding set is incomplete")
    if "created_at" not in run_context:
        run_context["created_at"] = _iso_now()
    if _parse_aware_timestamp(run_context.get("created_at")) is None:
        raise ValueError("owned run context has no aware created_at")
    source_identities = {
        (
            row.get("source_commit"),
            row.get("source_tree"),
            row.get("direct_base_url"),
            row.get("gateway_base_url"),
            row.get("direct_health_url"),
            row.get("gateway_health_url"),
            row.get("cdp_url"),
            row.get("electron_pid"),
            row.get("gateway_pid"),
        )
        for row in expected_binding.values()
        if isinstance(row, dict)
    }
    if len(source_identities) != 1:
        raise ValueError("owned representatives do not share one runtime source")
    plans = {
        name: {
            **spec,
            "env": dict(spec.get("env") or {}),
        }
        for name, spec in plans.items()
    }
    primary_expected = expected_binding[V5_PRIMARY_REPRESENTATIVE_ID]
    run_intent = _v5_build_run_intent(run_context, expected_binding)
    run_intent_path = _v5_owned_coordination_path(run_dir, "run-intent.json")
    run_intent_bytes = _v5_write_exclusive_json(
        run_intent_path,
        run_intent,
    )
    run_intent_sha256 = hashlib.sha256(run_intent_bytes).hexdigest()
    phase_paths = [
        _v5_phase_coordination_paths(run_dir, phase)
        for phase in V5_CACHE_PHASES
    ]
    cache_artifact_root = _v5_owned_coordination_path(
        run_dir,
        "cache-live-artifacts",
    )
    base_replacements = {
        "{PHASE_CONTROL_DIR}": str(run_dir),
        "{CACHE_ARTIFACT_ROOT}": str(cache_artifact_root),
        "{SOURCE_COMMIT}": str(primary_expected["source_commit"]),
        "{SOURCE_TREE}": str(primary_expected["source_tree"]),
        "{RUN_INTENT_PATH}": str(run_intent_path),
        "{RUN_INTENT_SHA256}": run_intent_sha256,
    }
    active_handles: list[dict[str, Any]] = []
    cache_handle: dict[str, Any] | None = None
    ui_results: list[dict[str, Any]] = []
    api_results: list[dict[str, Any]] = []
    cache_result: dict[str, Any] | None = None
    phase_bindings: list[dict[str, Any]] = []
    private_attestation_token = secrets.token_urlsafe(48)
    private_attestation_token_path, private_attestation_token_fd = (
        _v5_open_exclusive_capture(
            run_dir,
            "private-cache-attestation.token",
        )
    )
    try:
        token_bytes = private_attestation_token.encode("ascii")
        if os.write(private_attestation_token_fd, token_bytes) != len(token_bytes):
            raise OSError("private attestation token write was incomplete")
        os.fsync(private_attestation_token_fd)
    except BaseException:
        private_attestation_token_path.unlink(missing_ok=True)
        raise
    finally:
        os.close(private_attestation_token_fd)
    plans["ui"]["env"][PRIVATE_CACHE_ATTESTATION_TOKEN_FILE_ENV] = str(
        private_attestation_token_path
    )
    plans["cache"]["env"][PRIVATE_CACHE_ATTESTATION_TOKEN_ENV] = (
        private_attestation_token
    )
    try:
        ready_timeout = float(
            plans["ui"].get("ready_timeout_seconds") or 1_800
        )
        representative_sessions: dict[str, str] = {}
        prior_backend_pid: int | None = None
        for phase, paths in zip(
            V5_CACHE_PHASES,
            phase_paths,
            strict=True,
        ):
            representative_id = phase["representative_id"]
            reuse_session_id = (
                representative_sessions.get(representative_id, "")
                if phase["index"] in {1, 2, 3, 4}
                else ""
            )
            phase_replacements = {
                **base_replacements,
                "{READY_PATH}": str(paths["ready"]),
                "{SESSION_BINDING_PATH}": str(paths["binding"]),
                "{RELEASE_PATH}": str(paths["release"]),
                "{PAIRED_API_PATH}": str(paths["paired_api"]),
                "{ACTIVE_PHASE_INDEX}": str(phase["index"]),
                "{UI_SESSION_ATTESTATION_PATH}": str(
                    paths["ui_attestation"]
                ),
                "{PREVIOUS_BACKEND_PID}": str(prior_backend_pid or 0),
                "{REUSE_SESSION_ID}": reuse_session_id,
                "{REUSE_SESSION_ATTESTATION_PATH}": (
                    str(phase_paths[phase["index"] - 1]["ui_attestation"])
                    if phase["index"] in {1, 2, 3, 4}
                    else ""
                ),
            }
            ui_handle = _v5_start_owned_child(
                "ui",
                plans["ui"],
                run_context,
                run_dir,
                replacements=phase_replacements,
                output_basename=f"ui.phase-{phase['index']:02d}.producer.json",
            )
            active_handles.append(ui_handle)
            binding, binding_digest, hold_observation = _v5_wait_for_ui_hold(
                ui_handle,
                paths["ready"],
                paths["binding"],
                run_context,
                expected_binding[phase["representative_id"]],
                phase,
                timeout=ready_timeout,
                observer=hold_observer,
            )
            if (
                binding["electron_pid"] != primary_expected["electron_pid"]
                or binding["gateway_pid"] != primary_expected["gateway_pid"]
                or binding["cdp_url"] != primary_expected["cdp_url"]
                or (
                    phase["restart_required"]
                    and (
                        binding.get("previous_backend_pid")
                        != prior_backend_pid
                        or binding["backend_pid"] == prior_backend_pid
                    )
                )
            ):
                raise RuntimeError(
                    "UI phase did not preserve its Electron/gateway identity "
                    "across a real backend restart"
                )
            session_id = str(binding["session_id"])
            prior_session = representative_sessions.get(representative_id)
            if prior_session is None:
                if (
                    representative_id == V5_NATIVE_REPRESENTATIVE_ID
                    and session_id
                    == representative_sessions.get(
                        V5_PRIMARY_REPRESENTATIVE_ID
                    )
                ):
                    raise RuntimeError(
                        "native representative reused the primary model session"
                    )
                representative_sessions[representative_id] = session_id
            elif session_id != prior_session:
                raise RuntimeError(
                    "UI phase changed session within one representative"
                )
            if phase["index"] in {0, 5} and reuse_session_id:
                raise RuntimeError("new-session phase unexpectedly reused a session")
            if phase["index"] in {1, 2, 3, 4} and (
                not reuse_session_id or session_id != reuse_session_id
            ):
                raise RuntimeError("primary phase did not reuse its exact UI session")
            prior_backend_pid = int(binding["backend_pid"])
            phase_bindings.append(
                {
                    "phase": phase,
                    "binding": binding,
                    "binding_sha256": binding_digest,
                    "hold_observation": hold_observation,
                }
            )
            if phase["index"] == 0:
                cache_handle = _v5_start_owned_child(
                    "cache",
                    plans["cache"],
                    run_context,
                    run_dir,
                    replacements=phase_replacements,
                    output_basename="cache.producer.json",
                )
                active_handles.append(cache_handle)
            if ui_handle["process"].poll() is not None:
                raise RuntimeError(
                    f"UI producer exited during cache phase {phase['name']}"
                )
            if cache_handle is None:
                raise RuntimeError("cache producer did not start with phase zero")
            done, done_bytes = _v5_wait_for_cache_phase_done(
                cache_handle,
                paths["cache_done"],
                run_context=run_context,
                phase=phase,
                binding=binding,
                binding_sha256=binding_digest,
                timeout=float(
                    plans["cache"].get("producer_timeout_seconds") or 7_200
                ),
            )
            if paths["release"].exists() or paths["release"].is_symlink():
                raise RuntimeError(
                    "UI phase release appeared before paired API proof completion"
                )
            api_handle = _v5_start_owned_child(
                "api",
                plans["api"],
                run_context,
                run_dir,
                replacements={
                    **phase_replacements,
                    "{REUSE_SESSION_ID}": "",
                    "{REUSE_SESSION_ATTESTATION_PATH}": "",
                },
                output_basename=f"api.phase-{phase['index']:02d}.producer.json",
            )
            active_handles.append(api_handle)
            api_result = _v5_finish_owned_child(
                api_handle,
                run_context,
                timeout=float(
                    plans["api"].get("producer_timeout_seconds") or 7_200
                ),
            )
            _v5_require_child_hold_binding(
                api_result,
                session_id=session_id,
                binding_sha256=binding_digest,
                phase=phase,
            )
            api_results.append(api_result)
            _, attestation_bytes = _v5_wait_for_ui_session_attestation(
                ui_handle,
                paths["ui_attestation"],
                run_context=run_context,
                run_intent_sha256=run_intent_sha256,
                phase=phase,
                binding=binding,
                timeout=ready_timeout,
            )
            _, api_matrix_bytes = _v5_read_owned_json(paths["paired_api"])
            release = {
                "schema": V5_UI_RELEASE_SCHEMA,
                "run_id": run_context["run_id"],
                "nonce": run_context["nonce"],
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": phase["representative_id"],
                "bundle_role": phase["bundle_role"],
                "cache_policy": phase["cache_policy"],
                "paged_ram": phase["paged_ram"],
                "model": binding["model"],
                "bundle_fingerprint_sha256": binding[
                    "bundle_fingerprint_sha256"
                ],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "session_id": binding["session_id"],
                "run_intent_sha256": run_intent_sha256,
                "ui_session_attestation_sha256": hashlib.sha256(
                    attestation_bytes
                ).hexdigest(),
                "api_capture_sha256": hashlib.sha256(
                    api_matrix_bytes
                ).hexdigest(),
                "cache_capture_sha256": hashlib.sha256(
                    done_bytes
                ).hexdigest(),
                "released_at": _iso_now(),
            }
            release_bytes = _v5_write_exclusive_json(
                paths["release"],
                release,
            )
            ui_result = _v5_finish_owned_child(
                ui_handle,
                run_context,
                timeout=float(
                    plans["ui"].get("producer_timeout_seconds") or 7_200
                ),
            )
            _v5_require_child_hold_binding(
                ui_result,
                session_id=session_id,
                binding_sha256=binding_digest,
                phase=phase,
            )
            if ui_result["capture"].get("release_sha256") != hashlib.sha256(
                release_bytes
            ).hexdigest():
                raise RuntimeError("UI producer consumed a different phase release")
            ui_results.append(ui_result)
        if cache_handle is None:
            raise RuntimeError("cache producer was never started")
        cache_result = _v5_finish_owned_child(
            cache_handle,
            run_context,
            timeout=float(
                plans["cache"].get("producer_timeout_seconds") or 7_200
            ),
        )
        _v5_require_multiphase_cache_binding(cache_result, phase_bindings)

        def aggregate_phase_results(
            name: str,
            rows: list[dict[str, Any]],
        ) -> dict[str, Any]:
            captures: list[str] = []
            for row in rows:
                envelope = row["capture"]
                encoded = envelope.get("captures")
                if not isinstance(encoded, list) or len(encoded) != 1:
                    raise RuntimeError(
                        f"{name} phase producer returned an invalid capture set"
                    )
                captures.extend(encoded)
            envelope = {
                "schema": V5_PRODUCER_ENVELOPE_SCHEMA,
                "producer": name,
                "run_id": run_context["run_id"],
                "nonce": run_context["nonce"],
                "captured_during_ui_hold": True,
                "captures": captures,
                "phase_count": len(rows),
            }
            return {
                "name": name,
                "executions": [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "capture"
                    }
                    for row in rows
                ],
                "capture": envelope,
                "capture_sha256": _canonical_json_sha256(envelope),
                "session_id": phase_bindings[-1]["binding"]["session_id"],
                "session_binding_sha256": phase_bindings[-1][
                    "binding_sha256"
                ],
                "hold_phases": phase_bindings,
            }

        ui_aggregate = aggregate_phase_results("ui", ui_results)
        api_aggregate = aggregate_phase_results("api", api_results)
        runtime = dict(phase_bindings[-1]["hold_observation"])
        runtime["phase_observations"] = [
            {
                "phase_index": row["phase"]["index"],
                "session_id": row["binding"]["session_id"],
                "backend_pid": row["binding"]["backend_pid"],
                "observation": row["hold_observation"],
            }
            for row in phase_bindings
        ]
        ui_aggregate["hold_observation"] = runtime
        api_aggregate["hold_observation"] = runtime
        cache_result["hold_observation"] = runtime
        cache_result["hold_phases"] = phase_bindings
        return {
            "ui": ui_aggregate,
            "api": api_aggregate,
            "cache": cache_result,
        }
    finally:
        for handle in active_handles:
            _v5_abort_owned_child(handle)
        private_attestation_token_path.unlink(missing_ok=True)


def _v5_worker_private_root() -> Path:
    value = os.environ.get("TMPDIR")
    if not value:
        raise RuntimeError("owned worker has no parent-provided TMPDIR")
    root = Path(value)
    before = root.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_nlink < 1
    ):
        raise RuntimeError("owned worker TMPDIR is unsafe")
    return root.resolve()


def _v5_worker_path(
    value: Path,
    root: Path,
    *,
    name: str,
    must_exist: bool,
) -> Path:
    lexical = Path(os.path.abspath(value))
    if lexical.parent != root or lexical.name != name:
        raise RuntimeError(f"owned worker {name} path is not parent-bound")
    exists = lexical.exists() or lexical.is_symlink()
    if exists != must_exist:
        state = "missing" if must_exist else "stale"
        raise RuntimeError(f"owned worker {name} path is {state}")
    if exists:
        observed = lexical.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"owned worker {name} path is a symlink")
    return lexical


def _v5_worker_optional_path(
    value: Path,
    root: Path,
    *,
    name: str,
    directory: bool = False,
) -> Path:
    lexical = Path(os.path.abspath(value))
    if lexical.parent != root or lexical.name != name:
        raise RuntimeError(f"owned worker {name} path is not parent-bound")
    if lexical.exists() or lexical.is_symlink():
        observed = lexical.lstat()
        valid = (
            stat.S_ISDIR(observed.st_mode)
            if directory
            else stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1
        )
        if stat.S_ISLNK(observed.st_mode) or not valid:
            raise RuntimeError(f"owned worker {name} path is unsafe")
    return lexical


def _v5_worker_source_and_bundle(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source = _v5_git_snapshot()
    if (
        source.get("commit") != args.v5_source_commit
        or source.get("tree") != args.v5_source_tree
    ):
        raise RuntimeError("owned worker source identity mismatch")
    primary = _read_bundle_directory_snapshot(args.bundle_root)
    native = _read_bundle_directory_snapshot(args.native_bundle_root)
    if (
        primary is None
        or native is None
        or primary["fingerprint_sha256"] == native["fingerprint_sha256"]
        or primary["model_bundle_path"] == native["model_bundle_path"]
        or native["derived"]["native_cache"]
        not in {
            "minimax_m3_sparse",
            "dsv4_composite",
            "openpangu_native",
            "cca",
        }
    ):
        raise RuntimeError("owned worker representative bundle identity is unsafe")
    return source, {
        V5_PRIMARY_REPRESENTATIVE_ID: primary,
        V5_NATIVE_REPRESENTATIVE_ID: native,
    }


def _v5_import_pinned_module(path: Path, name: str) -> tuple[Any, dict[str, Any]]:
    pin = _v5_pin_regular_file(path)
    spec = importlib.util.spec_from_file_location(name, pin["path"])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source-owned harness: {path.name}")
    module = importlib.util.module_from_spec(spec)
    # Match normal import semantics before executing the module.  Python 3.13's
    # dataclass decorator resolves annotations through sys.modules while the
    # class body is being processed, so an unregistered dynamic module fails
    # before the source-owned API harness can run.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not _v5_pin_unchanged(pin):
        sys.modules.pop(name, None)
        raise RuntimeError(f"source-owned harness changed while loading: {path.name}")
    return module, pin


def _v5_protocol_output_tokens(protocol: str, raw: bytes) -> int:
    maximum = 0
    try:
        whole = json.loads(raw)
        if isinstance(whole, dict):
            if protocol == "chat":
                usage = whole.get("usage")
                candidate = (
                    usage.get("completion_tokens")
                    if isinstance(usage, dict)
                    else 0
                )
            elif protocol in {"responses", "anthropic"}:
                usage = whole.get("usage")
                candidate = (
                    usage.get("output_tokens")
                    if isinstance(usage, dict)
                    else 0
                )
            else:
                candidate = whole.get("eval_count")
            if isinstance(candidate, int):
                maximum = max(maximum, candidate)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        for raw_line in raw.decode("utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if protocol != "ollama":
                if not line.startswith("data:"):
                    continue
                line = line.removeprefix("data:").strip()
                if line == "[DONE]":
                    continue
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            if protocol == "chat":
                usage = value.get("usage")
                candidate = usage.get("completion_tokens") if isinstance(
                    usage,
                    dict,
                ) else 0
            elif protocol == "responses":
                response = value.get("response")
                usage = response.get("usage") if isinstance(response, dict) else {}
                candidate = usage.get("output_tokens") if isinstance(
                    usage,
                    dict,
                ) else 0
            elif protocol == "anthropic":
                usage = value.get("usage")
                candidate = usage.get("output_tokens") if isinstance(
                    usage,
                    dict,
                ) else 0
            else:
                candidate = value.get("eval_count")
            if isinstance(candidate, int):
                maximum = max(maximum, candidate)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return maximum
    return maximum


def _v5_reasoning_mode_request(
    protocol: str,
    payload: dict[str, Any],
    capture_label: str,
) -> tuple[dict[str, Any], str]:
    match = re.search(r"round([123])$", capture_label)
    if match is None:
        raise RuntimeError("unexpected API capture label")
    stage = int(match.group(1))
    mode = ("auto", "on", "off")[stage - 1]
    request = json.loads(json.dumps(payload))
    key = "think" if protocol == "ollama" else "enable_thinking"
    request.pop("enable_thinking", None)
    request.pop("think", None)
    if mode == "on":
        request[key] = True
    elif mode == "off":
        request[key] = False
    return request, mode


def _v5_parse_resolved_sampling_log(
    line: str,
    *,
    route: str,
    expected_models: set[str],
    proof_request_id: str,
    request_id: str = "",
    message_id: str = "",
) -> dict[str, Any] | None:
    marker = f"Resolved sampling kwargs route={route} model="
    start = line.find(marker)
    kwargs_marker = line.find(" kwargs=", start + len(marker))
    if start < 0 or kwargs_marker < 0:
        return None
    model_and_ids = line[start + len(marker) : kwargs_marker]
    identity_matches = list(
        re.finditer(
            r" (proof_request_id|request_id|message_id)="
            r"([A-Za-z0-9_.:-]{1,160})",
            model_and_ids,
        )
    )
    if not identity_matches:
        return None
    observed_model = model_and_ids[: identity_matches[0].start()]
    expected_position = identity_matches[0].start()
    identities: dict[str, str] = {}
    for match in identity_matches:
        if match.start() != expected_position or match.group(1) in identities:
            return None
        identities[match.group(1)] = match.group(2)
        expected_position = match.end()
    if expected_position != len(model_and_ids):
        return None
    observed_proof_id = identities.get("proof_request_id", "")
    if observed_model not in expected_models:
        return None
    if observed_proof_id != proof_request_id:
        return None
    if request_id and identities.get("request_id") != request_id:
        return None
    if message_id and identities.get("message_id") != message_id:
        return None
    raw = line[kwargs_marker + len(" kwargs=") :].strip()
    values: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "max_tokens",
        "enable_thinking",
    ):
        match = re.search(rf"['\"]{re.escape(key)}['\"]\s*:\s*([^,}}]+)", raw)
        if match is None:
            continue
        token = match.group(1).strip().strip("'\"")
        if token.lower() in {"true", "false"}:
            values[key] = token.lower() == "true"
        elif token.lower() in {"none", "null"}:
            values[key] = None
        else:
            try:
                number = float(token)
            except ValueError:
                values[key] = token
            else:
                values[key] = int(number) if number.is_integer() else number
    if not values:
        return None
    return {
        "route": route,
        "model": observed_model,
        "proof_request_id": observed_proof_id,
        "request_id": identities.get("request_id", ""),
        "message_id": identities.get("message_id", ""),
        "values": values,
        "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
        "line_b64": _v5_encode_bytes(line.encode()),
    }


def _v5_wait_for_resolved_sampling_log(
    binding: dict[str, Any],
    before: list[str],
    *,
    route: str,
    proof_request_id: str,
    request_id: str = "",
    message_id: str = "",
    timeout: float = 15.0,
) -> tuple[dict[str, Any], list[str]]:
    deadline = time.monotonic() + timeout
    expected_models = {
        str(binding["model"]),
        str(binding["model_bundle_path"]),
    }
    if any(
        _v5_parse_resolved_sampling_log(
            row,
            route=route,
            expected_models=expected_models,
            proof_request_id=proof_request_id,
            request_id=request_id,
            message_id=message_id,
        )
        is not None
        for row in before
    ):
        raise RuntimeError("sampling probe proof ID existed before request")
    while time.monotonic() < deadline:
        current = _v5_cdp_session_logs(
            str(binding["cdp_url"]),
            str(binding["session_id"]),
        )
        observations = [
            parsed
            for row in current
            if (
                parsed := _v5_parse_resolved_sampling_log(
                    row,
                    route=route,
                    expected_models=expected_models,
                    proof_request_id=proof_request_id,
                    request_id=request_id,
                    message_id=message_id,
                )
            )
            is not None
        ]
        if len(observations) == 1:
            return observations[0], current
        if len(observations) > 1:
            raise RuntimeError("sampling probe observed multiple resolved requests")
        time.sleep(0.1)
    raise RuntimeError("sampling probe observed no resolved server kwargs")


def _v5_api_sampling_capture(
    harness: Any,
    original_send: Any,
    binding: dict[str, Any],
) -> dict[str, Any]:
    health_before_bytes = _v5_loopback_http_get(str(binding["health_url"]))
    health_before = json.loads(health_before_bytes)
    effective_defaults = health_before.get("effective_defaults")
    if not isinstance(effective_defaults, dict) or not effective_defaults:
        raise RuntimeError("sampling probe health has no effective defaults")
    default_temperature = effective_defaults.get("temperature")
    override_temperature = (
        0.123
        if not isinstance(default_temperature, (int, float))
        or abs(float(default_temperature) - 0.123) > 1e-6
        else 0.456
    )
    client = harness.ProtocolClient(
        str(binding["direct_base_url"]),
        None,
        300,
        base_label="direct",
    )
    observations: list[dict[str, Any]] = []
    for label, override in (
        ("default", {}),
        ("override", {"temperature": override_temperature}),
        ("after_override", {}),
    ):
        proof_request_id = f"r18-sampling-{label}-{secrets.token_hex(12)}"
        request_id = f"{proof_request_id}-request"
        message_id = f"{proof_request_id}-message"
        request = {
            "model": binding["model"],
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply exactly R18-SAMPLING-{label.upper()}.",
                }
            ],
            "stream": False,
            "max_tokens": 16,
            "enable_thinking": False,
            **override,
        }
        logs_before = _v5_cdp_session_logs(
            str(binding["cdp_url"]),
            str(binding["session_id"]),
        )
        client.headers["x-vmlx-proof-request-id"] = proof_request_id
        client.headers["x-vmlx-request-id"] = request_id
        client.headers["x-vmlx-message-id"] = message_id
        result = original_send(
            client,
            "chat",
            request,
            False,
            capture_label=f"sampling_{label}",
        )
        if (
            result.get("status_code") != 200
            or result.get("errors")
            or not result.get("terminals")
        ):
            raise RuntimeError("sampling probe request did not complete successfully")
        resolved, logs_after = _v5_wait_for_resolved_sampling_log(
            binding,
            logs_before,
            route="/v1/chat/completions",
            proof_request_id=proof_request_id,
            request_id=request_id,
            message_id=message_id,
        )
        request_bytes = json.dumps(
            request,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        result_bytes = json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        observations.append(
            {
                "label": label,
                "proof_request_id": proof_request_id,
                "request_id": request_id,
                "message_id": message_id,
                "request_b64": _v5_encode_bytes(request_bytes),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "result_b64": _v5_encode_bytes(result_bytes),
                "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "log_start_index": len(logs_before),
                "log_end_index": len(logs_after),
                "resolved": resolved,
            }
        )
    health_after_bytes = _v5_loopback_http_get(str(binding["health_url"]))
    health_after = json.loads(health_after_bytes)
    if health_after.get("effective_defaults") != effective_defaults:
        raise RuntimeError("per-request sampling override changed server defaults")
    return {
        "schema": "vmlx-r18-owned-sampling-attestation-v1",
        "health_effective_defaults": effective_defaults,
        "health_before_sha256": hashlib.sha256(health_before_bytes).hexdigest(),
        "health_after_sha256": hashlib.sha256(health_after_bytes).hexdigest(),
        "default_resolved": observations[0]["resolved"]["values"],
        "override_request": {"temperature": override_temperature},
        "override_resolved": observations[1]["resolved"]["values"],
        "after_override_resolved": observations[2]["resolved"]["values"],
        "observations": observations,
    }


def _v5_api_worker_capture(
    args: argparse.Namespace,
    binding: dict[str, Any],
    run_root: Path,
) -> bytes:
    harness_path = ROOT / "tests/cross_matrix/run_agentic_protocol_matrix.py"
    harness, harness_pin = _v5_import_pinned_module(
        harness_path,
        f"_vmlx_r18_api_worker_{args.v5_nonce}",
    )
    raw_root = (
        run_root
        / f"api-parser-input-phase-{args.v5_active_phase_index:02d}"
    )
    output_path = args.v5_paired_api_path
    records: list[dict[str, Any]] = []
    original_send = harness.ProtocolClient.send

    def recorded_send(
        client: Any,
        protocol: str,
        payload: dict[str, Any],
        stream: bool,
        *,
        capture_label: str = "request",
    ) -> dict[str, Any]:
        request, reasoning_mode = _v5_reasoning_mode_request(
            protocol,
            payload,
            capture_label,
        )
        started_ns = time.monotonic_ns()
        result = original_send(
            client,
            protocol,
            request,
            stream,
            capture_label=capture_label,
        )
        ended_ns = time.monotonic_ns()
        records.append(
            {
                "protocol": protocol,
                "route": client.base_label,
                "capture_label": capture_label,
                "reasoning_mode": reasoning_mode,
                "stream": bool(stream),
                "request": request,
                "started_ns": started_ns,
                "ended_ns": ended_ns,
            }
        )
        return result

    harness.ProtocolClient.send = recorded_send
    phase = V5_CACHE_PHASES[args.v5_active_phase_index]
    full_agentic = phase["api_action_profile"] in {
        "full-agentic-plus-cache-store",
        "full-agentic-native-cache",
    }
    selected_protocols = None if full_agentic else ["chat"]
    selected_modes = ["stream", "nonstream"] if full_agentic else ["stream"]
    matrix_args = argparse.Namespace(
        base_url=[
            f"direct={binding['direct_base_url']}",
            f"gateway={binding['gateway_base_url']}",
        ],
        health_url=[
            f"direct={binding['health_url']}",
            f"gateway={binding['gateway_health_url']}",
        ],
        model=binding["model"],
        bundle_root=Path(binding["model_bundle_path"]),
        repo_root=str(ROOT),
        output=output_path,
        raw_artifact_dir=raw_root,
        source_head=binding["source_commit"],
        run_id=args.v5_run_id,
        api_key=None,
        protocol=selected_protocols,
        mode=selected_modes,
        max_output_tokens=1024,
        recovery_max_tokens=128,
        timeout=300,
        minimum_abort_deltas=3,
        disconnect_delay_ms=1_000,
        enable_thinking=True,
        second_tool_choice="explicit",
        skip_cancellation=True,
        allow_single_base=False,
    )
    try:
        matrix = harness.run_matrix(matrix_args)
    finally:
        harness.ProtocolClient.send = original_send
    sampling = _v5_api_sampling_capture(
        harness,
        original_send,
        binding,
    )
    harness._write_private_result_exclusive(
        output_path,
        matrix,
        ROOT,
    )
    if matrix.get("pass") is not True:
        raise RuntimeError("source-owned raw API matrix did not pass")
    raw_capture = matrix.get("raw_capture")
    if not isinstance(raw_capture, dict) or raw_capture.get("complete") is not True:
        raise RuntimeError("source-owned raw API matrix capture is incomplete")
    raw_dir = raw_root / args.v5_run_id
    artifact_by_key: dict[tuple[str, str, str], tuple[bytes, dict[str, Any]]] = {}
    for route in raw_capture.get("routes") or []:
        if not isinstance(route, dict):
            continue
        key = (
            str(route.get("base_label") or ""),
            str(route.get("protocol") or ""),
            str(route.get("capture_label") or ""),
        )
        artifacts = route.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise RuntimeError("raw API route does not have exactly one artifact")
        artifact = artifacts[0]
        body_path = raw_dir / str(artifact.get("body_file") or "")
        metadata_path = raw_dir / str(artifact.get("metadata_file") or "")
        body = body_path.read_bytes()
        metadata = json.loads(metadata_path.read_bytes())
        if (
            hashlib.sha256(body).hexdigest() != artifact.get("body_sha256")
            or not isinstance(metadata, dict)
        ):
            raise RuntimeError("raw API parser-input artifact changed")
        artifact_by_key[key] = body, metadata
    flows: list[dict[str, Any]] = []
    endpoint_by_protocol = {
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
        "ollama": "/api/chat",
    }
    for record in records:
        key = (
            record["route"],
            record["protocol"],
            record["capture_label"],
        )
        if key not in artifact_by_key:
            raise RuntimeError("raw API parser-input route is missing")
        response, metadata = artifact_by_key[key]
        request_bytes = json.dumps(
            record["request"],
            allow_nan=False,
        ).encode("utf-8")
        request_metadata = metadata.get("request")
        response_metadata = metadata.get("response")
        if (
            not isinstance(request_metadata, dict)
            or not isinstance(response_metadata, dict)
            or request_metadata.get("body_sha256")
            != hashlib.sha256(request_bytes).hexdigest()
        ):
            raise RuntimeError("raw API request bytes do not match transport metadata")
        first_byte_ms = response_metadata.get("first_byte_ms")
        completed_ms = response_metadata.get("completed_ms")
        output_tokens = _v5_protocol_output_tokens(
            record["protocol"],
            response,
        )
        if (
            not isinstance(first_byte_ms, (int, float))
            or not isinstance(completed_ms, (int, float))
            or first_byte_ms <= 0
            or completed_ms <= first_byte_ms
            or output_tokens <= 0
        ):
            raise RuntimeError("raw API timing or usage evidence is incomplete")
        started_ns = int(record["started_ns"])
        first_ns = started_ns + int(float(first_byte_ms) * 1_000_000)
        ended_ns = started_ns + int(float(completed_ms) * 1_000_000)
        timing = {
            "started_ns": started_ns,
            "first_byte_ns": first_ns,
            "ended_ns": ended_ns,
            "output_tokens": output_tokens,
            "displayed_ttft_ms": (first_ns - started_ns) / 1_000_000,
            "displayed_tps": output_tokens
            / ((ended_ns - first_ns) / 1_000_000_000),
        }
        flows.append(
            {
                "protocol": record["protocol"],
                "route": record["route"],
                "endpoint": endpoint_by_protocol[record["protocol"]],
                "mode": "stream" if record["stream"] else "nonstream",
                "reasoning_mode": record["reasoning_mode"],
                "request_b64": _v5_encode_bytes(request_bytes),
                "response_b64": _v5_encode_bytes(response),
                "timing_b64": _v5_encode_bytes(
                    json.dumps(
                        timing,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ),
            }
        )
    expected_flow_count = len(records)
    if (
        expected_flow_count <= 0
        or len(flows) != expected_flow_count
        or not _v5_pin_unchanged(harness_pin)
    ):
        raise RuntimeError(
            "raw API worker did not retain its exact phase-specific matrix"
        )
    capture = {
        "schema": V5_API_SCHEMA,
        "run_id": args.v5_run_id,
        "nonce": args.v5_nonce,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "api_action_profile": phase["api_action_profile"],
        "session_id": binding["session_id"],
        "session_binding_sha256": hashlib.sha256(
            args.v5_session_binding_path.read_bytes()
        ).hexdigest(),
        "flows": flows,
        "sampling_b64": _v5_encode_bytes(
            json.dumps(
                sampling,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "matrix_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    return json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()


def _v5_prepare_fixed_node_path(
    run_root: Path,
    *,
    phase_index: int,
) -> tuple[Path, list[dict[str, Any]]]:
    candidates = (
        (Path("/opt/homebrew/bin/node"), Path("/opt/homebrew/bin/npm")),
        (Path("/usr/local/bin/node"), Path("/usr/local/bin/npm")),
    )
    for node_link, npm_link in candidates:
        try:
            node = node_link.resolve(strict=True)
            npm = npm_link.resolve(strict=True)
            node, npm, bin_dir, tool_pins = _v5_prepare_node_toolchain(
                run_root,
                node_path=node,
                npm_cli_path=npm,
                bin_name=f"fixed-node-bin-phase-{phase_index:02d}",
            )
        except (OSError, ValueError):
            continue
        return node, tool_pins
    raise RuntimeError("no pinned Node/npm toolchain is available")


def _v5_wait_for_single_file(
    directory: Path,
    pattern: str,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = sorted(directory.glob(pattern))
        if len(matches) > 1:
            raise RuntimeError(f"UI harness produced duplicate {pattern} files")
        if len(matches) == 1:
            return matches[0]
        if process.poll() is not None:
            raise RuntimeError(f"UI harness exited before producing {pattern}")
        time.sleep(0.1)
    raise RuntimeError(f"UI harness timed out before producing {pattern}")


def _v5_ui_normalized_capture(
    args: argparse.Namespace,
    proof: dict[str, Any],
    proof_bytes: bytes,
    session_id: str,
) -> bytes:
    active_phase = V5_CACHE_PHASES[args.v5_active_phase_index]
    turn_count = int(active_phase["ui_turn_count"])
    if (
        proof.get("format") != ELECTRON_PROOF_SCHEMA
        or proof.get("run_id") != args.v5_run_id
        or nested(proof, "session", "id") != session_id
        or nested(proof, "uiStartControl", "clicked") is not True
    ):
        raise RuntimeError("source-owned UI proof is not bound to the held session")
    prompts = [
        nested(proof, "requestContract", f"prompt{label}")
        for label in ("One", "Two", "Three")[:turn_count]
    ]
    assistant_ids = proof.get("assistantMessageIds")
    trace_rows = proof.get("messageEventTrace")
    if (
        any(not isinstance(prompt, str) or not prompt for prompt in prompts)
        or not isinstance(assistant_ids, list)
        or len(assistant_ids) < turn_count
        or not isinstance(trace_rows, list)
    ):
        raise RuntimeError("source-owned UI proof lacks its exact phase turn traces")
    traces = {
        str(row.get("messageId") or ""): row.get("events")
        for row in trace_rows
        if isinstance(row, dict) and isinstance(row.get("events"), list)
    }
    persisted_calls = proof.get("persistedOaiCallsByMessage")
    persisted_results = proof.get("persistedOaiResultsByMessage")
    if (
        not isinstance(persisted_calls, list)
        or len(persisted_calls) < turn_count
        or not isinstance(persisted_results, list)
        or len(persisted_results) < turn_count
    ):
        raise RuntimeError("source-owned UI proof lacks persisted tool records")
    turns: list[dict[str, Any]] = []
    for index, message_id in enumerate(assistant_ids[:turn_count]):
        raw_events = traces.get(str(message_id))
        if not isinstance(raw_events, list) or not raw_events:
            raise RuntimeError("source-owned UI proof is missing an event trace")
        call_rows = persisted_calls[index]
        result_rows = persisted_results[index]
        if not isinstance(call_rows, list) or not isinstance(result_rows, list):
            raise RuntimeError("source-owned UI tool records are malformed")
        call_by_id = {
            str(row.get("id") or row.get("toolCallId") or ""): row
            for row in call_rows
            if isinstance(row, dict)
        }
        result_by_id = {
            str(
                row.get("tool_call_id")
                or row.get("toolCallId")
                or row.get("callId")
                or ""
            ): row
            for row in result_rows
            if isinstance(row, dict)
        }
        normalized: list[dict[str, Any]] = []
        for raw_event in sorted(
            raw_events,
            key=lambda row: int(row.get("sequence") or 0)
            if isinstance(row, dict)
            else 0,
        ):
            if not isinstance(raw_event, dict):
                raise RuntimeError("source-owned UI event trace is malformed")
            event = str(raw_event.get("event") or "")
            payload = raw_event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            channel = str(raw_event.get("channel") or "")
            if event == "stream":
                delta = raw_event.get("delta")
                if isinstance(delta, str) and delta:
                    normalized.append(
                        {
                            "type": (
                                "reasoning_delta"
                                if channel == "reasoning"
                                else "content_delta"
                            ),
                            "text": delta,
                        }
                    )
            elif event == "tool":
                phase = str(
                    payload.get("phase")
                    or payload.get("status")
                    or payload.get("type")
                    or ""
                ).lower()
                call_id = str(
                    payload.get("toolCallId")
                    or payload.get("tool_call_id")
                    or payload.get("callId")
                    or payload.get("id")
                    or ""
                )
                persisted_call = call_by_id.get(call_id)
                persisted_function = (
                    persisted_call.get("function")
                    if isinstance(persisted_call, dict)
                    and isinstance(persisted_call.get("function"), dict)
                    else {}
                )
                name = str(
                    persisted_function.get("name")
                    or payload.get("toolName")
                    or payload.get("name")
                    or ""
                )
                arguments = (
                    persisted_function.get("arguments")
                    or payload.get("arguments")
                    or {}
                )
                if isinstance(arguments, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        arguments = json.loads(arguments)
                if any(token in phase for token in ("result", "complete", "done")):
                    persisted_result = result_by_id.get(call_id)
                    content = (
                        (
                            persisted_result.get("content")
                            if isinstance(persisted_result, dict)
                            else None
                        )
                        or payload.get("content")
                        or payload.get("result")
                        or payload.get("output")
                        or payload.get("detail")
                        or ""
                    )
                    normalized.append(
                        {
                            "type": "tool_result",
                            "call_id": call_id,
                            "content": str(content),
                        }
                    )
                elif call_id and name and isinstance(arguments, dict):
                    normalized.append(
                        {
                            "type": "tool_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
            elif event == "terminal":
                metrics = payload.get("metrics")
                metrics = metrics if isinstance(metrics, dict) else {}
                ttft_ms = (
                    metrics.get("ttft_ms")
                    or metrics.get("ttftMs")
                    or payload.get("ttft_ms")
                    or payload.get("ttftMs")
                )
                if ttft_ms is None and metrics.get("ttft") is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        ttft_ms = float(metrics["ttft"]) * 1_000
                tps: Any = (
                    metrics.get("tokens_per_second")
                    or metrics.get("decode_tps")
                    or metrics.get("tokensPerSecond")
                    or payload.get("decode_tps")
                )
                with contextlib.suppress(TypeError, ValueError):
                    tps = float(tps)
                finish_reason = str(payload.get("finishReason") or "").lower()
                if (
                    not isinstance(ttft_ms, (int, float))
                    or not isinstance(tps, (int, float))
                    or finish_reason
                    not in {"stop", "end_turn", "completed", "complete"}
                ):
                    raise RuntimeError("source-owned UI terminal has no raw timing")
                normalized.append(
                    {
                        "type": "terminal",
                        "status": "completed",
                        "response_id": str(payload.get("responseId") or ""),
                        "ttft_ms": float(ttft_ms),
                        "decode_tps": float(tps),
                    }
                )
        if not _successful_terminal(
            [{**row, "seq": sequence} for sequence, row in enumerate(normalized)]
        ):
            raise RuntimeError("source-owned UI trace has no successful terminal")
        normalized = [
            {"seq": sequence, **row}
            for sequence, row in enumerate(normalized)
        ]
        request = {
            "messages": [{"role": "user", "content": prompts[index]}],
            "message_id": str(message_id),
        }
        turns.append(
            {
                "request_b64": _v5_encode_bytes(
                    json.dumps(
                        request,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ),
                "events_b64": _v5_encode_bytes(
                    b"".join(
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                        + b"\n"
                        for row in normalized
                    )
                ),
            }
        )
    interaction = [
        {
            "method": "Runtime.evaluate.visibleHTMLElement.click",
            "selector": "button[data-action='start-session']",
            "session_id": session_id,
            "label": nested(proof, "uiStartControl", "label"),
            "status_before": nested(
                proof,
                "uiStartControl",
                "sessionStatusBefore",
            ),
            "status_after": nested(
                proof,
                "uiStartControl",
                "sessionStatusAfter",
            ),
        }
    ]
    capture = {
        "schema": V5_UI_SCHEMA,
        "run_id": args.v5_run_id,
        "nonce": args.v5_nonce,
        "phase_index": active_phase["index"],
        "phase_name": active_phase["name"],
        "representative_id": active_phase["representative_id"],
        "ui_action_profile": active_phase["ui_action_profile"],
        "ui_turn_count": turn_count,
        "session_id": session_id,
        "source_proof_b64": _v5_encode_bytes(proof_bytes),
        "interaction_b64": _v5_encode_bytes(
            json.dumps(
                interaction,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "turns": turns,
    }
    return json.dumps(capture, sort_keys=True, separators=(",", ":")).encode()


def _v5_ui_worker_capture(
    args: argparse.Namespace,
    source: dict[str, Any],
    bundle: dict[str, Any],
    run_root: Path,
) -> tuple[bytes, dict[str, Any], bytes]:
    phase = V5_CACHE_PHASES[args.v5_active_phase_index]
    active_model = (
        args.model
        if phase["representative_id"] == V5_PRIMARY_REPRESENTATIVE_ID
        else args.native_model
    )
    node, tool_pins = _v5_prepare_fixed_node_path(
        run_root,
        phase_index=args.v5_active_phase_index,
    )
    harness_path = ROOT / "panel/scripts/live-real-ui-model-proof.mjs"
    harness_pin = _v5_pin_regular_file(harness_path)
    proof_dir = run_root / f"ui-live-proof-phase-{phase['index']:02d}"
    proof_dir.mkdir(mode=0o700)
    stdout_path, stdout_fd = _v5_open_exclusive_capture(
        run_root,
        f"ui-harness.phase-{phase['index']:02d}.stdout",
    )
    stderr_path, stderr_fd = _v5_open_exclusive_capture(
        run_root,
        f"ui-harness.phase-{phase['index']:02d}.stderr",
    )
    direct_port = _v5_host_port(args.direct_base_url)[1]
    fixed_bin = (
        run_root
        / f"fixed-node-bin-phase-{args.v5_active_phase_index:02d}"
    )
    private_attestation_token_file = os.environ.get(
        PRIVATE_CACHE_ATTESTATION_TOKEN_FILE_ENV,
        "",
    ).strip()
    if not private_attestation_token_file:
        raise RuntimeError("UI proof worker has no private attestation token file")
    environment = _v5_minimal_env(
        run_root,
        {
            PRIVATE_CACHE_ATTESTATION_TOKEN_FILE_ENV: (
                private_attestation_token_file
            ),
        },
    )
    environment.update(
        {
            "PATH": f"{fixed_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "VMLINUX_REAL_UI_MODEL_PATH": str(bundle["model_bundle_path"]),
            "VMLINUX_REAL_UI_SERVED_MODEL": active_model,
            "VMLINUX_REAL_UI_PROOF_DIR": str(proof_dir),
            "VMLINUX_REAL_UI_RUN_ID": args.v5_run_id,
            "VMLINUX_REAL_UI_PROOF_BASENAME": (
                f"v5-owned-ui-phase-{phase['index']:02d}"
            ),
            "VMLINUX_REAL_UI_SERVER_PORT": str(direct_port),
            "VMLINUX_REAL_UI_PAIRED_API_HOLD_SECONDS": "7200",
            "VMLINUX_REAL_UI_PAIRED_API_ARTIFACT": str(
                args.v5_paired_api_path
            ),
            "VMLINUX_REAL_UI_PAIRED_CACHE_ARTIFACT": str(
                _v5_existing_phase_paths(
                    args.v5_phase_control_dir,
                    phase,
                )["cache_done"]
            ),
            "VMLINUX_REAL_UI_RELEASE_SENTINEL": str(args.v5_release_path),
            "VMLINUX_REAL_UI_NONCE": args.v5_nonce,
            "VMLINUX_REAL_UI_RUN_INTENT_PATH": str(args.v5_run_intent_path),
            "VMLINUX_REAL_UI_RUN_INTENT_SHA256": args.v5_run_intent_sha256,
            "VMLINUX_REAL_UI_ACTIVE_PHASE_INDEX": str(phase["index"]),
            "VMLINUX_REAL_UI_SESSION_ATTESTATION_PATH": str(
                args.v5_ui_session_attestation_path
            ),
            "VMLINUX_REAL_UI_GATEWAY_PID": str(args.gateway_pid),
            "VMLINUX_REAL_UI_GATEWAY_BASE_URL": args.gateway_base_url,
            "VMLINUX_REAL_UI_ATTACH_CDP_URL": args.cdp_url,
            "VMLINUX_REAL_UI_EXPECTED_ELECTRON_PID": str(args.electron_pid),
            "VMLINUX_REAL_UI_LIFECYCLE_OWNER": "parent",
            "VMLINUX_REAL_UI_ALLOW_TEARDOWN": "0",
            "VMLINUX_REAL_UI_REUSE_SESSION_ID": args.v5_reuse_session_id,
            "VMLINUX_REAL_UI_REUSE_SESSION_ATTESTATION_PATH": (
                str(args.v5_reuse_session_attestation_path)
                if args.v5_reuse_session_id
                else ""
            ),
            "VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS": "1",
            "VMLINUX_REAL_UI_EXPECT_PAGED_CACHE": (
                "1" if phase["paged_ram"] else "0"
            ),
            "VMLINUX_REAL_UI_MAX_TOKENS": "2048",
            "VMLINUX_REAL_UI_BUILTIN_TOOLS": "1",
            "VMLINUX_REAL_UI_ALLOW_FAIL": "1",
        }
    )
    process = subprocess.Popen(  # noqa: S603 - pinned source-owned harness
        [str(node), str(harness_path)],
        cwd=ROOT / "panel",
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stdout_fd,
        stderr=stderr_fd,
    )
    os.close(stdout_fd)
    os.close(stderr_fd)
    try:
        harness_binding_path = _v5_wait_for_single_file(
            proof_dir,
            "*-ui-backend-binding.json",
            process,
            timeout=1_800,
        )
        harness_binding, harness_binding_bytes = _v5_read_owned_json(
            harness_binding_path
        )
        cdp_url = str(harness_binding.get("cdp_url") or "")
        electron_pid = int(harness_binding.get("electron_pid") or 0)
        backend_pid = int(harness_binding.get("backend_pid") or 0)
        session_id = str(harness_binding.get("local_session_id") or "")
        if (
            harness_binding.get("run_id") != args.v5_run_id
            or harness_binding.get("source_commit") != source["commit"]
            or harness_binding.get("source_tree") != source["tree"]
            or harness_binding.get("base_url") != args.direct_base_url
            or harness_binding.get("served_model") != active_model
            or harness_binding.get("model_bundle_fingerprint_sha256")
            != bundle["fingerprint_sha256"]
            or not session_id
            or backend_pid <= 0
            or electron_pid <= 0
            or not cdp_url
        ):
            raise RuntimeError("source-owned UI harness binding is incomplete")
        binding = {
            "schema": V5_SESSION_BINDING_SCHEMA,
            "run_id": args.v5_run_id,
            "nonce": args.v5_nonce,
            "ui_producer_pid": os.getpid(),
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "model": active_model,
            "model_bundle_path": bundle["model_bundle_path"],
            "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
            "session_id": session_id,
            "direct_base_url": args.direct_base_url,
            "gateway_base_url": args.gateway_base_url,
            "health_url": args.health_url,
            "direct_health_url": args.health_url,
            "gateway_health_url": args.gateway_health_url,
            "cdp_url": cdp_url,
            "backend_pid": backend_pid,
            "gateway_pid": int(args.gateway_pid),
            "electron_pid": electron_pid,
            "harness_binding_sha256": hashlib.sha256(
                harness_binding_bytes
            ).hexdigest(),
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "bundle_role": phase["bundle_role"],
            "cache_policy": phase["cache_policy"],
            "kv_cache_quantization": phase["kv_cache_quantization"],
            "tq_policy": phase["tq_policy"],
            "session_policy": phase["session_policy"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
            "paged_ram": phase["paged_ram"],
            "session_start_ordinal": phase["index"] + 1,
            "previous_backend_pid": (
                args.v5_previous_backend_pid
                if args.v5_previous_backend_pid > 0
                else None
            ),
        }
        binding_bytes = _v5_write_exclusive_json(
            args.v5_session_binding_path,
            binding,
        )
        ready = {
            "schema": V5_UI_READY_SCHEMA,
            "run_id": args.v5_run_id,
            "nonce": args.v5_nonce,
            "ui_producer_pid": os.getpid(),
            "session_id": session_id,
            "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "held": True,
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "bundle_role": phase["bundle_role"],
            "cache_policy": phase["cache_policy"],
            "kv_cache_quantization": phase["kv_cache_quantization"],
            "tq_policy": phase["tq_policy"],
            "session_policy": phase["session_policy"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
            "paged_ram": phase["paged_ram"],
            "ready_at": _iso_now(),
        }
        _v5_write_exclusive_json(args.v5_ready_path, ready)
        try:
            process.wait(timeout=7_200)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise RuntimeError("source-owned UI harness hold timed out") from exc
        if not args.v5_release_path.exists():
            raise RuntimeError("source-owned UI harness exited before parent release")
        if process.returncode != 0:
            raise RuntimeError("source-owned UI harness failed")
        proof_path = _v5_wait_for_single_file(
            proof_dir,
            "*-proof.json",
            process,
            timeout=1,
        )
        proof, proof_bytes = _v5_read_owned_json(proof_path)
        release, release_bytes = _v5_read_owned_json(args.v5_release_path)
        attestation, _ = _v5_read_owned_json(
            args.v5_ui_session_attestation_path
        )
        _v5_validate_ui_session_attestation(
            attestation,
            run_context={
                "run_id": args.v5_run_id,
                "nonce": args.v5_nonce,
            },
            run_intent_sha256=args.v5_run_intent_sha256,
            phase=phase,
            binding=binding,
        )
        expected_release = {
            "schema": V5_UI_RELEASE_SCHEMA,
            "run_id": args.v5_run_id,
            "nonce": args.v5_nonce,
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "bundle_role": phase["bundle_role"],
            "cache_policy": phase["cache_policy"],
            "paged_ram": phase["paged_ram"],
            "model": active_model,
            "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
            "session_id": session_id,
            "run_intent_sha256": args.v5_run_intent_sha256,
            "ui_session_attestation_sha256": hashlib.sha256(
                args.v5_ui_session_attestation_path.read_bytes()
            ).hexdigest(),
            "api_capture_sha256": hashlib.sha256(
                args.v5_paired_api_path.read_bytes()
            ).hexdigest(),
            "cache_capture_sha256": hashlib.sha256(
                _v5_existing_phase_paths(
                    args.v5_phase_control_dir,
                    phase,
                )["cache_done"].read_bytes()
            ).hexdigest(),
        }
        if (
            set(release) != {*expected_release, "released_at"}
            or any(
                release.get(key) != value
                for key, value in expected_release.items()
            )
            or _parse_aware_timestamp(release.get("released_at")) is None
        ):
            raise RuntimeError("source-owned UI harness consumed wrong release")
        capture = _v5_ui_normalized_capture(
            args,
            proof,
            proof_bytes,
            session_id,
        )
        if (
            not _v5_pin_unchanged(harness_pin)
            or any(
                not _v5_pin_unchanged(
                    pin,
                    executable=pin["path"] != str(Path("/bin/sh").resolve()),
                )
                for pin in tool_pins
            )
        ):
            raise RuntimeError("UI harness or Node/npm toolchain changed")
        return capture, binding, release_bytes
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        # Log bytes remain private and hashed by the parent worker envelope;
        # they are never copied into the public manifest.
        for path in (stdout_path, stderr_path):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("UI harness log capture was replaced")


def _v5_paged_mode_from_cache_summary(summary: dict[str, Any]) -> bool:
    topology = nested(
        summary,
        "identity",
        "cache_topology_provenance",
        "configuration",
    )
    candidates = (
        topology.get("paged_cache") if isinstance(topology, dict) else None,
        topology.get("paged_cache_enabled")
        if isinstance(topology, dict)
        else None,
        nested(topology, "configured", "paged_cache"),
        nested(topology, "configured", "paged_cache_enabled"),
        nested(topology, "configured", "use_paged_cache"),
        nested(topology, "instantiated", "paged_cache"),
        nested(topology, "instantiated", "paged_cache_enabled"),
        nested(topology, "instantiated", "paged_ram_enabled"),
        nested(
            summary,
            "health_before",
            "cache",
            "scheduler_cache",
            "paged_cache",
            "paged_ram_enabled",
        ),
    )
    values = [value for value in candidates if isinstance(value, bool)]
    if not values or any(value != values[0] for value in values[1:]):
        raise RuntimeError("cache gate does not attest one unambiguous paged mode")
    return values[0]


def _v5_cache_worker_phase(
    args: argparse.Namespace,
    binding: dict[str, Any],
    run_root: Path,
    phase: dict[str, Any],
    *,
    store_summary_path: Path | None,
) -> tuple[dict[str, Any], Path, bytes]:
    """Run one source-owned cache store/probe phase against its held UI engine."""

    gate_operation = _v5_cache_gate_operation(phase)
    harness_path = ROOT / "tests/cross_matrix/run_cache_hierarchy_live_gate.py"
    harness_pin = _v5_pin_regular_file(harness_path)
    python_path = (ROOT / ".venv/bin/python").resolve(strict=True)
    python_pin = _v5_pin_regular_file(python_path, executable=True)
    artifact_root = args.v5_cache_artifact_root
    phase_root = artifact_root / f"{phase['index']:02d}-{phase['name']}"
    phase_root.mkdir(mode=0o700)
    command = [
        str(python_path),
        str(harness_path),
        "--base-url",
        str(binding["direct_base_url"]),
        "--model",
        str(binding["model"]),
        "--nonce",
        args.v5_nonce,
        "--source-identity",
        args.v5_source_commit,
        "--config-identity",
        str(binding["bundle_fingerprint_sha256"]),
        "--artifact-dir",
        str(phase_root),
        "--phase",
        gate_operation,
        "--cache-scenario",
        _v5_cache_gate_scenario(phase),
    ]
    if gate_operation == "probe":
        if store_summary_path is None:
            raise RuntimeError("cache probe phase has no linked store summary")
        command.extend(("--store-summary", str(store_summary_path)))
    elif store_summary_path is not None:
        raise RuntimeError("cache store phase unexpectedly names an old summary")
    private_attestation_token = os.environ.get(
        PRIVATE_CACHE_ATTESTATION_TOKEN_ENV,
        "",
    ).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,512}", private_attestation_token):
        raise RuntimeError("cache proof worker has no valid attestation credential")
    completed = subprocess.run(  # noqa: S603 - pinned source-owned harness
        command,
        cwd=ROOT,
        env=_v5_minimal_env(
            run_root,
            {PRIVATE_CACHE_ATTESTATION_TOKEN_ENV: private_attestation_token},
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=3_600,
        check=False,
    )
    if (
        completed.returncode != 0
        or not _v5_pin_unchanged(harness_pin)
        or not _v5_pin_unchanged(python_pin, executable=True)
    ):
        raise RuntimeError("source-owned cache gate failed or changed")
    summary_path = phase_root / "summary.json"
    summary, summary_bytes = _v5_read_owned_json(summary_path)
    identity = summary.get("identity")
    observed_engine = (
        identity.get("observed_engine") if isinstance(identity, dict) else None
    )
    observed_bundle = (
        identity.get("model_bundle_provenance")
        if isinstance(identity, dict)
        else None
    )
    if (
        summary.get("schema") != CACHE_PROOF_SCHEMA
        or summary.get("phase") != gate_operation
        or summary.get("cache_scenario") != _v5_cache_gate_scenario(phase)
        or summary.get("nonce") != args.v5_nonce
        or summary.get("base_url") != binding["direct_base_url"]
        or summary.get("model") != binding["model"]
        or summary.get("gate_ok") is not True
        or not isinstance(observed_engine, dict)
        or observed_engine.get("pid") != binding["backend_pid"]
        or not isinstance(observed_bundle, dict)
        or observed_bundle.get("fingerprint_sha256")
        != binding["bundle_fingerprint_sha256"]
        or _v5_paged_mode_from_cache_summary(summary) != phase["paged_ram"]
    ):
        raise RuntimeError("cache gate summary is not bound to the held session")
    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(phase_root.rglob("*")):
        if path.is_dir():
            continue
        observed = path.lstat()
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
        ):
            raise RuntimeError("cache gate produced an unsafe artifact")
        artifact_rows.append(
            {
                "relative_path": path.relative_to(phase_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": observed.st_size,
            }
        )
    if not artifact_rows:
        raise RuntimeError("cache gate produced no retained artifacts")
    linked_store_sha256 = (
        hashlib.sha256(store_summary_path.read_bytes()).hexdigest()
        if store_summary_path is not None
        else None
    )
    scenario = {
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "kv_cache_quantization": phase["kv_cache_quantization"],
        "tq_policy": phase["tq_policy"],
        "session_policy": phase["session_policy"],
        "operation": phase["operation"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "paged_ram": _v5_paged_mode_from_cache_summary(summary),
        "model": binding["model"],
        "bundle_fingerprint_sha256": binding["bundle_fingerprint_sha256"],
        "session_id": binding["session_id"],
        "backend_pid": binding["backend_pid"],
        "session_binding_sha256": binding["__binding_sha256"],
        "summary_b64": _v5_encode_bytes(summary_bytes),
        "linked_store_summary_sha256": linked_store_sha256,
        "artifact_manifest_b64": _v5_encode_bytes(
            json.dumps(
                artifact_rows,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
    }
    return scenario, summary_path, summary_bytes


def _v5_wait_for_worker_phase_binding(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    phase: dict[str, Any],
    *,
    timeout: float,
) -> tuple[dict[str, Any], bytes, bytes]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if (
            args.v5_session_binding_path.exists()
            or args.v5_session_binding_path.is_symlink()
        ) and (
            args.v5_ready_path.exists()
            or args.v5_ready_path.is_symlink()
        ):
            try:
                return _v5_worker_session_binding(args, bundle, phase)
            except (OSError, RuntimeError) as exc:
                last_error = exc
        time.sleep(0.05)
    if last_error is not None:
        raise RuntimeError(
            f"cache worker phase {phase['name']} binding stayed invalid"
        ) from last_error
    raise RuntimeError(f"cache worker phase {phase['name']} binding timed out")


def _v5_wait_for_phase_release(
    args: argparse.Namespace,
    phase: dict[str, Any],
    binding: dict[str, Any],
    binding_bytes: bytes,
    done_bytes: bytes,
    *,
    timeout: float,
) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if args.v5_release_path.exists() or args.v5_release_path.is_symlink():
            release, release_bytes = _v5_read_owned_json(args.v5_release_path)
            expected = {
                "schema": V5_UI_RELEASE_SCHEMA,
                "run_id": args.v5_run_id,
                "nonce": args.v5_nonce,
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": phase["representative_id"],
                "bundle_role": phase["bundle_role"],
                "cache_policy": phase["cache_policy"],
                "paged_ram": phase["paged_ram"],
                "model": binding["model"],
                "bundle_fingerprint_sha256": binding[
                    "bundle_fingerprint_sha256"
                ],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "session_id": binding["session_id"],
                "run_intent_sha256": args.v5_run_intent_sha256,
                "ui_session_attestation_sha256": hashlib.sha256(
                    _v5_existing_phase_paths(
                        args.v5_phase_control_dir,
                        phase,
                    )["ui_attestation"].read_bytes()
                ).hexdigest(),
                "api_capture_sha256": hashlib.sha256(
                    _v5_existing_phase_paths(
                        args.v5_phase_control_dir,
                        phase,
                    )["paired_api"].read_bytes()
                ).hexdigest(),
                "cache_capture_sha256": hashlib.sha256(
                    done_bytes
                ).hexdigest(),
            }
            if (
                set(release) != {*expected, "released_at"}
                or any(
                    release.get(key) != value
                    for key, value in expected.items()
                )
                or _parse_aware_timestamp(release.get("released_at")) is None
            ):
                raise RuntimeError("cache worker observed wrong phase release")
            return release_bytes
        time.sleep(0.05)
    raise RuntimeError(f"cache worker phase {phase['name']} release timed out")


def _v5_cache_worker_capture(
    args: argparse.Namespace,
    bundles: dict[str, dict[str, Any]],
    run_root: Path,
) -> tuple[bytes, list[dict[str, Any]], dict[str, Any], bytes]:
    """Run six real policy/cache phases across parent-controlled UI restarts."""

    args.v5_cache_artifact_root.mkdir(mode=0o700)
    phases: list[dict[str, Any]] = []
    phase_bindings: list[dict[str, Any]] = []
    store_summaries: dict[tuple[str, bool, str], Path] = {}
    last_binding: dict[str, Any] | None = None
    last_binding_bytes = b""
    for phase in V5_CACHE_PHASES:
        gate_operation = _v5_cache_gate_operation(phase)
        paths = _v5_existing_phase_paths(args.v5_phase_control_dir, phase)
        phase_previous_backend_pid = (
            int(last_binding["backend_pid"])
            if last_binding is not None
            else int(args.v5_previous_backend_pid)
        )
        phase_args = argparse.Namespace(
            **{
                **vars(args),
                "v5_session_binding_path": paths["binding"],
                "v5_ready_path": paths["ready"],
                "v5_release_path": paths["release"],
                "v5_previous_backend_pid": phase_previous_backend_pid,
            }
        )
        binding, binding_bytes, _ = _v5_wait_for_worker_phase_binding(
            phase_args,
            bundles[phase["representative_id"]],
            phase,
            timeout=7_200,
        )
        binding_sha256 = hashlib.sha256(binding_bytes).hexdigest()
        binding["__binding_sha256"] = binding_sha256
        store_key = (
            phase["representative_id"],
            bool(phase["paged_ram"]),
            phase["cache_policy"],
        )
        store_path = (
            store_summaries.get(store_key)
            if gate_operation == "probe"
            else None
        )
        scenario, summary_path, summary_bytes = _v5_cache_worker_phase(
            phase_args,
            binding,
            run_root,
            phase,
            store_summary_path=store_path,
        )
        if gate_operation == "store":
            store_summaries[store_key] = summary_path
        done = {
            "schema": V5_CACHE_PHASE_DONE_SCHEMA,
            "run_id": args.v5_run_id,
            "nonce": args.v5_nonce,
            "phase_index": phase["index"],
            "phase_name": phase["name"],
            "representative_id": phase["representative_id"],
            "bundle_role": phase["bundle_role"],
            "cache_policy": phase["cache_policy"],
            "kv_cache_quantization": phase["kv_cache_quantization"],
            "tq_policy": phase["tq_policy"],
            "session_policy": phase["session_policy"],
            "operation": phase["operation"],
            "ui_action_profile": phase["ui_action_profile"],
            "ui_turn_count": phase["ui_turn_count"],
            "api_action_profile": phase["api_action_profile"],
            "paged_ram": phase["paged_ram"],
            "session_id": binding["session_id"],
            "backend_pid": binding["backend_pid"],
            "session_binding_sha256": binding_sha256,
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "completed_at": _iso_now(),
        }
        done_bytes = _v5_write_exclusive_json(paths["cache_done"], done)
        phases.append(scenario)
        phase_bindings.append(
            {
                "phase_index": phase["index"],
                "phase_name": phase["name"],
                "representative_id": phase["representative_id"],
                "cache_policy": phase["cache_policy"],
                "kv_cache_quantization": phase["kv_cache_quantization"],
                "tq_policy": phase["tq_policy"],
                "session_policy": phase["session_policy"],
                "ui_action_profile": phase["ui_action_profile"],
                "ui_turn_count": phase["ui_turn_count"],
                "api_action_profile": phase["api_action_profile"],
                "session_id": binding["session_id"],
                "model": binding["model"],
                "bundle_fingerprint_sha256": binding[
                    "bundle_fingerprint_sha256"
                ],
                "backend_pid": binding["backend_pid"],
                "session_binding_sha256": binding_sha256,
            }
        )
        last_binding = binding
        last_binding_bytes = binding_bytes
        if phase["index"] != len(V5_CACHE_PHASES) - 1:
            _v5_wait_for_phase_release(
                phase_args,
                phase,
                binding,
                binding_bytes,
                done_bytes,
                timeout=7_200,
            )
    if last_binding is None:
        raise RuntimeError("cache worker completed no phases")
    phase2_summary, phase2_summary_bytes = _v5_json_bytes(
        phases[2].get("summary_b64")
    )
    phase3_summary, phase3_summary_bytes = _v5_json_bytes(
        phases[3].get("summary_b64")
    )
    if not isinstance(phase2_summary, dict) or not isinstance(
        phase3_summary,
        dict,
    ):
        raise RuntimeError("cache worker lacks L2 phase summaries")
    try:
        l2_attestation = _v5_derive_l2_size_eviction_attestation(
            run_id=args.v5_run_id,
            nonce=args.v5_nonce,
            phase2_summary=phase2_summary,
            phase2_summary_sha256=hashlib.sha256(
                phase2_summary_bytes
            ).hexdigest(),
            phase3_summary=phase3_summary,
            phase3_summary_sha256=hashlib.sha256(
                phase3_summary_bytes
            ).hexdigest(),
        )
    except ValueError as exc:
        raise RuntimeError(
            "cache worker lacks strict L2 size/LRU/restart evidence"
        ) from exc
    capture = {
        "schema": V5_CACHE_SCHEMA,
        "run_id": args.v5_run_id,
        "nonce": args.v5_nonce,
        "session_id": last_binding["session_id"],
        "phases": phases,
        "l2_size_eviction_attestation": l2_attestation,
    }
    return (
        json.dumps(capture, sort_keys=True, separators=(",", ":")).encode(),
        phase_bindings,
        last_binding,
        last_binding_bytes,
    )


def _v5_worker_output_fd(descriptor: int) -> os.stat_result:
    if descriptor < 3:
        raise RuntimeError("owned worker output descriptor is unsafe")
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_size != 0
        or stat.S_IMODE(observed.st_mode) & 0o077
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("owned worker output descriptor is not an empty private file")
    return observed


def _v5_worker_session_binding(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    phase: dict[str, Any],
) -> tuple[dict[str, Any], bytes, bytes]:
    if args.v5_release_path.exists() or args.v5_release_path.is_symlink():
        raise RuntimeError("owned worker session was already released")
    binding, binding_bytes = _v5_read_owned_json(
        args.v5_session_binding_path
    )
    ready, ready_bytes = _v5_read_owned_json(args.v5_ready_path)
    binding_sha256 = hashlib.sha256(binding_bytes).hexdigest()
    expected_model = (
        args.model
        if phase["representative_id"] == V5_PRIMARY_REPRESENTATIVE_ID
        else args.native_model
    )
    required_binding = {
        "schema": V5_SESSION_BINDING_SCHEMA,
        "run_id": args.v5_run_id,
        "nonce": args.v5_nonce,
        "source_commit": args.v5_source_commit,
        "source_tree": args.v5_source_tree,
        "model": expected_model,
        "model_bundle_path": bundle["model_bundle_path"],
        "bundle_fingerprint_sha256": bundle["fingerprint_sha256"],
        "direct_base_url": args.direct_base_url,
        "gateway_base_url": args.gateway_base_url,
        "health_url": args.health_url,
        "direct_health_url": args.health_url,
        "gateway_health_url": args.gateway_health_url,
        "cdp_url": args.cdp_url,
        "electron_pid": args.electron_pid,
        "gateway_pid": args.gateway_pid,
        "session_start_ordinal": phase["index"] + 1,
        "previous_backend_pid": (
            args.v5_previous_backend_pid
            if args.v5_previous_backend_pid > 0
            else None
        ),
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "kv_cache_quantization": phase["kv_cache_quantization"],
        "tq_policy": phase["tq_policy"],
        "session_policy": phase["session_policy"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "paged_ram": phase["paged_ram"],
    }
    required_ready = {
        "schema": V5_UI_READY_SCHEMA,
        "run_id": args.v5_run_id,
        "nonce": args.v5_nonce,
        "ui_producer_pid": binding.get("ui_producer_pid"),
        "session_id": binding.get("session_id"),
        "binding_sha256": binding_sha256,
        "held": True,
        "phase_index": phase["index"],
        "phase_name": phase["name"],
        "representative_id": phase["representative_id"],
        "bundle_role": phase["bundle_role"],
        "cache_policy": phase["cache_policy"],
        "kv_cache_quantization": phase["kv_cache_quantization"],
        "tq_policy": phase["tq_policy"],
        "session_policy": phase["session_policy"],
        "ui_action_profile": phase["ui_action_profile"],
        "ui_turn_count": phase["ui_turn_count"],
        "api_action_profile": phase["api_action_profile"],
        "paged_ram": phase["paged_ram"],
    }
    if (
        any(binding.get(key) != value for key, value in required_binding.items())
        or any(ready.get(key) != value for key, value in required_ready.items())
    ):
        raise RuntimeError("owned worker session binding mismatch")
    ready_at = _parse_aware_timestamp(ready.get("ready_at"))
    now = datetime.now(UTC)
    if (
        ready_at is None
        or ready_at > now
        or (now - ready_at).total_seconds() > 7_200
    ):
        raise RuntimeError("owned worker ready sentinel is stale")
    ui_pid = binding.get("ui_producer_pid")
    if not isinstance(ui_pid, int) or not isinstance(_observe_process(ui_pid), dict):
        raise RuntimeError("owned worker UI producer is no longer alive")
    return binding, binding_bytes, ready_bytes


def _v5_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Private child worker for the vMLX V5 release preflight",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--v5-owned-worker",
        choices=V5_PRODUCER_NAMES,
        required=True,
    )
    parser.add_argument("--v5-output-fd", type=int, required=True)
    parser.add_argument("--v5-run-id", required=True)
    parser.add_argument("--v5-nonce", required=True)
    parser.add_argument("--v5-session-binding-path", type=Path, required=True)
    parser.add_argument("--v5-ready-path", type=Path, required=True)
    parser.add_argument("--v5-release-path", type=Path, required=True)
    parser.add_argument("--v5-phase-control-dir", type=Path)
    parser.add_argument("--v5-paired-api-path", type=Path, required=True)
    parser.add_argument("--v5-cache-artifact-root", type=Path, required=True)
    parser.add_argument("--v5-source-commit", required=True)
    parser.add_argument("--v5-source-tree", required=True)
    parser.add_argument("--v5-run-intent-path", type=Path, required=True)
    parser.add_argument("--v5-run-intent-sha256", required=True)
    parser.add_argument("--v5-active-phase-index", type=int, required=True)
    parser.add_argument(
        "--v5-ui-session-attestation-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--v5-previous-backend-pid", type=int, required=True)
    parser.add_argument("--v5-reuse-session-id", required=True)
    parser.add_argument("--v5-reuse-session-attestation-path", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--native-bundle-root", type=Path, required=True)
    parser.add_argument("--direct-base-url", required=True)
    parser.add_argument("--gateway-base-url", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--gateway-health-url", required=True)
    parser.add_argument("--cdp-url", required=True)
    parser.add_argument("--electron-pid", type=int, required=True)
    parser.add_argument("--gateway-pid", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-model", required=True)
    return parser


def _v5_write_worker_envelope(
    descriptor: int,
    envelope: dict[str, Any],
) -> None:
    payload = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("owned worker could not write its envelope")
        view = view[written:]
    os.fsync(descriptor)


def _v5_owned_worker_main(argv: list[str]) -> int:
    args = _v5_worker_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", args.v5_run_id):
        raise RuntimeError("owned worker run ID is unsafe")
    if not re.fullmatch(r"[0-9a-f]{32}", args.v5_nonce):
        raise RuntimeError("owned worker nonce is invalid")
    for value in (args.v5_source_commit, args.v5_source_tree):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise RuntimeError("owned worker source identity is invalid")
    _v5_worker_output_fd(args.v5_output_fd)
    run_root = _v5_worker_private_root()
    producer = args.v5_owned_worker
    require_existing = producer != "ui"
    if not 0 <= args.v5_active_phase_index < len(V5_CACHE_PHASES):
        raise RuntimeError("owned worker active phase index is invalid")
    active_phase = V5_CACHE_PHASES[args.v5_active_phase_index]
    if (
        producer == "cache"
        and args.v5_active_phase_index != V5_CACHE_PHASES[0]["index"]
    ):
        raise RuntimeError("multiphase cache worker must begin at phase zero")
    reuse_required = producer == "ui" and active_phase["index"] in {1, 2, 3, 4}
    if bool(args.v5_reuse_session_id) != reuse_required:
        raise RuntimeError("owned UI worker session-reuse policy is invalid")
    if bool(args.v5_reuse_session_attestation_path) != reuse_required:
        raise RuntimeError("owned UI worker reuse attestation policy is invalid")
    if args.v5_phase_control_dir is None:
        raise RuntimeError("owned worker has no phase-control directory")
    phase_control_dir = args.v5_phase_control_dir.resolve()
    if phase_control_dir != run_root:
        raise RuntimeError("owned worker phase-control directory is not its run root")
    args.v5_phase_control_dir = phase_control_dir
    active_phase_paths = _v5_existing_phase_paths(
        phase_control_dir,
        active_phase,
    )
    args.v5_session_binding_path = _v5_worker_path(
        args.v5_session_binding_path,
        run_root,
        name=active_phase_paths["binding"].name,
        must_exist=require_existing,
    )
    args.v5_ready_path = _v5_worker_path(
        args.v5_ready_path,
        run_root,
        name=active_phase_paths["ready"].name,
        must_exist=require_existing,
    )
    args.v5_release_path = _v5_worker_path(
        args.v5_release_path,
        run_root,
        name=active_phase_paths["release"].name,
        must_exist=False,
    )
    args.v5_paired_api_path = _v5_worker_path(
        args.v5_paired_api_path,
        run_root,
        name=active_phase_paths["paired_api"].name,
        must_exist=False,
    )
    args.v5_ui_session_attestation_path = (
        _v5_worker_path(
            args.v5_ui_session_attestation_path,
            run_root,
            name=active_phase_paths["ui_attestation"].name,
            must_exist=False,
        )
        if producer == "ui"
        else _v5_worker_optional_path(
            args.v5_ui_session_attestation_path,
            run_root,
            name=active_phase_paths["ui_attestation"].name,
        )
    )
    if reuse_required:
        prior_phase = V5_CACHE_PHASES[active_phase["index"] - 1]
        args.v5_reuse_session_attestation_path = _v5_worker_path(
            Path(args.v5_reuse_session_attestation_path),
            run_root,
            name=_v5_existing_phase_paths(
                phase_control_dir,
                prior_phase,
            )["ui_attestation"].name,
            must_exist=True,
        )
    args.v5_cache_artifact_root = (
        _v5_worker_path(
            args.v5_cache_artifact_root,
            run_root,
            name="cache-live-artifacts",
            must_exist=False,
        )
        if producer == "cache"
        else _v5_worker_optional_path(
            args.v5_cache_artifact_root,
            run_root,
            name="cache-live-artifacts",
            directory=True,
        )
    )
    args.v5_run_intent_path = _v5_worker_path(
        args.v5_run_intent_path,
        run_root,
        name="run-intent.json",
        must_exist=True,
    )
    source, bundles = _v5_worker_source_and_bundle(args)
    run_intent, run_intent_bytes = _v5_read_owned_json(
        args.v5_run_intent_path
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", args.v5_run_intent_sha256)
        or hashlib.sha256(run_intent_bytes).hexdigest()
        != args.v5_run_intent_sha256
    ):
        raise RuntimeError("owned worker run-intent bytes do not match")
    common_expected = {
        "source_commit": args.v5_source_commit,
        "source_tree": args.v5_source_tree,
        "direct_base_url": args.direct_base_url,
        "gateway_base_url": args.gateway_base_url,
        "health_url": args.health_url,
        "direct_health_url": args.health_url,
        "gateway_health_url": args.gateway_health_url,
        "cdp_url": args.cdp_url,
        "electron_pid": args.electron_pid,
        "gateway_pid": args.gateway_pid,
    }
    worker_representatives = {
        V5_PRIMARY_REPRESENTATIVE_ID: {
            **common_expected,
            "model": args.model,
            "model_bundle_path": bundles[V5_PRIMARY_REPRESENTATIVE_ID][
                "model_bundle_path"
            ],
            "bundle_fingerprint_sha256": bundles[
                V5_PRIMARY_REPRESENTATIVE_ID
            ]["fingerprint_sha256"],
            "native_cache_policy": bundles[V5_PRIMARY_REPRESENTATIVE_ID][
                "derived"
            ]["native_cache"],
        },
        V5_NATIVE_REPRESENTATIVE_ID: {
            **common_expected,
            "model": args.native_model,
            "model_bundle_path": bundles[V5_NATIVE_REPRESENTATIVE_ID][
                "model_bundle_path"
            ],
            "bundle_fingerprint_sha256": bundles[
                V5_NATIVE_REPRESENTATIVE_ID
            ]["fingerprint_sha256"],
            "native_cache_policy": bundles[V5_NATIVE_REPRESENTATIVE_ID][
                "derived"
            ]["native_cache"],
        },
    }
    _v5_validate_run_intent(
        run_intent,
        {
            "run_id": args.v5_run_id,
            "nonce": args.v5_nonce,
            "created_at": run_intent.get("created_at"),
        },
        worker_representatives,
    )
    active_bundle = bundles[active_phase["representative_id"]]
    if producer == "ui":
        (
            capture_bytes,
            binding,
            release_bytes,
        ) = _v5_ui_worker_capture(
            args,
            source,
            active_bundle,
            run_root,
        )
        binding_bytes = args.v5_session_binding_path.read_bytes()
        release_sha256 = hashlib.sha256(release_bytes).hexdigest()
        phase_releases = None
        cache_phase_bindings = None
    elif producer == "api":
        binding, binding_bytes, ready_bytes = _v5_worker_session_binding(
            args,
            active_bundle,
            active_phase,
        )
        capture_bytes = _v5_api_worker_capture(args, binding, run_root)
        rebound, rebound_bytes, rebound_ready_bytes = _v5_worker_session_binding(
            args,
            active_bundle,
            active_phase,
        )
        if (
            rebound != binding
            or rebound_bytes != binding_bytes
            or rebound_ready_bytes != ready_bytes
            or args.v5_release_path.exists()
            or args.v5_release_path.is_symlink()
        ):
            raise RuntimeError("owned worker session changed during capture")
        release_sha256 = None
        phase_releases = None
        cache_phase_bindings = None
    else:
        (
            capture_bytes,
            cache_phase_bindings,
            binding,
            binding_bytes,
        ) = _v5_cache_worker_capture(
            args,
            bundles,
            run_root,
        )
        release_sha256 = None
        phase_releases = None
    try:
        capture = json.loads(capture_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("owned worker normalized capture is malformed") from exc
    if (
        not isinstance(capture, dict)
        or capture.get("run_id") != args.v5_run_id
        or capture.get("nonce") != args.v5_nonce
    ):
        raise RuntimeError("owned worker normalized capture binding mismatch")
    envelope: dict[str, Any] = {
        "schema": V5_PRODUCER_ENVELOPE_SCHEMA,
        "producer": producer,
        "run_id": args.v5_run_id,
        "nonce": args.v5_nonce,
        "session_id": binding["session_id"],
        "session_binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "captured_during_ui_hold": True,
        "captures": [_v5_encode_bytes(capture_bytes)],
    }
    if producer in {"ui", "api"}:
        envelope.update(
            {
                "phase_index": active_phase["index"],
                "phase_name": active_phase["name"],
                "representative_id": active_phase["representative_id"],
                "ui_action_profile": active_phase["ui_action_profile"],
                "ui_turn_count": active_phase["ui_turn_count"],
                "api_action_profile": active_phase["api_action_profile"],
            }
        )
    if release_sha256 is not None:
        envelope["release_sha256"] = release_sha256
    if phase_releases is not None:
        envelope["phase_releases"] = phase_releases
    if cache_phase_bindings is not None:
        envelope["phase_bindings"] = cache_phase_bindings
    _v5_write_worker_envelope(args.v5_output_fd, envelope)
    if sys.stdin.buffer.readline() != b"finish\n":
        raise RuntimeError("owned worker did not receive parent completion")
    return 0


def _v5_validate_release_checks(
    *,
    before_source: dict[str, Any],
    after_source: dict[str, Any],
    runtime: dict[str, Any],
    representatives: dict[str, dict[str, Any]],
    owned_checks: dict[str, dict[str, Any]],
    captures: dict[str, list[tuple[dict[str, Any], bytes]]],
) -> tuple[dict[str, Any], list[str]]:
    if set(representatives) != set(V5_REPRESENTATIVE_IDS):
        raise RuntimeError("release representative set is incomplete")
    primary = representatives[V5_PRIMARY_REPRESENTATIVE_ID]
    bundle_snapshot = primary.get("bundle")
    if not isinstance(bundle_snapshot, dict):
        raise RuntimeError("primary representative bundle is missing")
    source_facts, scope_facts = _v5_source_and_scope_facts(
        before_source,
        after_source,
        runtime,
        bundle_snapshot,
    )
    ui_facts, ui_hashes = _v5_ui_facts(
        captures["ui"],
        runtime,
        bundle_snapshot,
    )
    api_by_protocol, api_facts, api_hashes = _v5_api_facts(
        captures["api"],
        bundle_snapshot,
    )
    cache_facts, cache_hashes = _v5_cache_facts(
        captures["cache"],
        representatives,
    )
    facts_by_check: dict[str, set[str]] = {
        "exact_source_provenance": source_facts,
        "release_scope_regression_review": scope_facts,
    }
    for name in OWNED_CHECK_NAMES:
        result = owned_checks.get(name)
        facts_by_check[name] = (
            set(result.get("facts") or []) if isinstance(result, dict) else set()
        )
    ui_mapping = {
        "real_start_button": "real_start_button",
        "minimum_three_turns": "minimum_three_turns",
        "reasoning_rail": "reasoning_rail_checked",
        "visible_content": "visible_content_checked",
        "tool_result_continuation": "tool_result_continuation_checked",
        "terminal_state": "terminal_state_checked",
        "cache_ttft_tps": "cache_ttft_tps_checked",
        "rendering": "rendering_checked",
        "coherence": "coherence_checked",
    }
    facts_by_check["electron_visual_multiturn"] = {
        assertion for fact, assertion in ui_mapping.items() if fact in ui_facts
    }
    api_mapping = {
        "reasoning_separate": "reasoning_content_separate",
        "content_progressive": "content_progressive",
        "tool_result_continuation": "tool_result_continuation_checked",
        "history_three_turn": "history_checked",
        "request_kwargs": "request_kwargs_checked",
        "terminal_truthful": "terminal_truthful",
    }
    for protocol in ("chat", "responses", "anthropic", "ollama"):
        facts = api_by_protocol.get(protocol, set())
        row = {
            assertion for fact, assertion in api_mapping.items() if fact in facts
        }
        if facts:
            row.update({"direct_stream_checked", "gateway_stream_checked"})
        facts_by_check[f"raw_api_{protocol}"] = row
    shared = ui_facts & api_facts
    facts_by_check["interleaved_reasoning_tools"] = {
        assertion
        for fact, assertion in {
            "reasoning_tool_reasoning_tool_answer": (
                "reasoning_tool_reasoning_tool_answer"
            ),
            "exact_tool_arguments": "exact_tool_arguments",
            "no_control_markup": "no_control_markup_leak",
            "nonempty_final": "nonempty_final_answer",
        }.items()
        if fact in shared
    }
    output_facts: set[str] = set()
    if "nonempty_final" in ui_facts & api_facts:
        output_facts.add("no_reasoning_only_finalization")
    if "no_stale_reasoning_replay" in ui_facts and "reasoning_not_stale" in api_facts:
        output_facts.add("no_stale_reasoning_replay")
    if "no_looping_or_gibberish" in ui_facts & api_facts:
        output_facts.add("no_looping_or_gibberish")
    if "terminal_state" in ui_facts and {
        "terminal_truthful",
        "success_finish_reasons",
    } <= api_facts:
        output_facts.update(
            {"no_eos_or_truncation_regression", "terminal_event_truthful"}
        )
    if "cache_ttft_tps" in ui_facts and "raw_timing_matches" in api_facts:
        output_facts.add("ttft_tps_compared_to_raw_timing")
    facts_by_check["output_integrity_terminal_stats"] = output_facts
    for name in (
        "cache_paged_off_ssd_partial",
        "cache_paged_on_eviction_refault",
        "cache_restart_and_size_eviction",
        "turboquant_policy",
    ):
        facts_by_check[name] = cache_facts & set(V5_RELEASE_ASSERTIONS[name])
    settings_facts = ui_facts | api_facts
    facts_by_check["settings_defaults_and_persistence"] = settings_facts & set(
        V5_RELEASE_ASSERTIONS["settings_defaults_and_persistence"]
    )
    parser_facts = api_facts.copy()
    if "reasoning_rail" in ui_facts and "no_control_markup" in ui_facts:
        parser_facts.add("no_inline_think_leak")
    facts_by_check["parser_family_matrix"] = parser_facts & set(
        V5_RELEASE_ASSERTIONS["parser_family_matrix"]
    )
    facts_by_check["i18n_katex_responsive_ui"] = ui_facts & set(
        V5_RELEASE_ASSERTIONS["i18n_katex_responsive_ui"]
    )
    evidence_by_check = {
        "electron_visual_multiturn": ui_hashes,
        "interleaved_reasoning_tools": sorted(set(ui_hashes + api_hashes)),
        "output_integrity_terminal_stats": sorted(set(ui_hashes + api_hashes)),
        "settings_defaults_and_persistence": sorted(set(ui_hashes + api_hashes)),
        "parser_family_matrix": sorted(set(ui_hashes + api_hashes)),
        "i18n_katex_responsive_ui": ui_hashes,
        **{
            f"raw_api_{protocol}": api_hashes
            for protocol in ("chat", "responses", "anthropic", "ollama")
        },
        **{
            name: cache_hashes
            for name in (
                "cache_paged_off_ssd_partial",
                "cache_paged_on_eviction_refault",
                "cache_restart_and_size_eviction",
                "turboquant_policy",
            )
        },
    }
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name in V5_REQUIRED_CHECKS:
        required = set(V5_RELEASE_ASSERTIONS[name])
        derived = facts_by_check.get(name, set()) & required
        assertions = {assertion: assertion in derived for assertion in required}
        status = "pass" if derived == required else "blocked"
        if status != "pass":
            failures.append(
                f"{name} blocked: missing {sorted(required - derived)}"
            )
        checks[name] = {
            "status": status,
            "assertions": dict(sorted(assertions.items())),
            "evidence_sha256": sorted(
                set(evidence_by_check.get(name, []))
            ),
        }
    return checks, failures


def _v5_manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = json.loads(json.dumps(manifest))
    completion = unsigned.get("completion")
    if isinstance(completion, dict):
        completion.pop("run_digest", None)
    return _canonical_json_sha256(unsigned)


def _v5_atomic_write_manifest(path: Path, manifest: dict[str, Any], nonce: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"manifest target already exists: {path}")
    temporary = path.parent / f".{path.name}.{nonce}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink()


def consume_v5_release_manifest(
    path: Path,
    *,
    expected_run_id: str,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    """Fail-closed contract used by downstream build/notary consumers."""

    root = path.expanduser().resolve().parent
    raw = _read_regular_file_once(path.expanduser().resolve(), root)
    if raw is None:
        raise ValueError("release manifest is not a stable regular file")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release manifest is not JSON") from exc
    checks = manifest.get("checks") if isinstance(manifest, dict) else None
    checks_complete = bool(
        isinstance(checks, dict)
        and set(checks) == set(V5_REQUIRED_CHECKS)
        and all(
            isinstance(row, dict)
            and row.get("status") == "pass"
            and set(row.get("assertions") or ())
            == set(V5_RELEASE_ASSERTIONS[name])
            and all(
                value is True
                for value in (row.get("assertions") or {}).values()
            )
            for name, row in checks.items()
        )
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != V5_MANIFEST_SCHEMA
        or manifest.get("status") != "pass"
        or nested(manifest, "run", "run_id") != expected_run_id
        or nested(manifest, "source", "commit") != expected_commit
        or nested(manifest, "source", "tree") != expected_tree
        or nested(manifest, "completion", "state") != "complete"
        or nested(manifest, "completion", "run_digest")
        != _v5_manifest_digest(manifest)
        or not checks_complete
    ):
        raise ValueError("release manifest is stale, incomplete, or not passing")
    return manifest


def _owned_command_plan(
    check_name: str,
    private_root: Path,
    run_id: str,
) -> list[dict[str, Any]] | None:
    python = (ROOT / ".venv/bin/python").resolve()
    npm = shutil.which("npm")
    if check_name == "full_python_suite" and python.is_file():
        return [
            {
                "command_id": check_name,
                "argv": [
                    str(python),
                    "-m",
                    "pytest",
                    "-s",
                    "-p",
                    "no:cacheprovider",
                ],
                "cwd": ROOT.resolve(),
                "env": {},
            }
        ]
    if check_name in {"full_panel_suite", "typecheck", "production_build"} and npm:
        suffix = {
            "full_panel_suite": ["test"],
            "typecheck": ["run", "typecheck"],
            "production_build": ["run", "build"],
        }[check_name]
        return [
            {
                "command_id": check_name,
                "argv": [str(Path(npm).resolve()), *suffix],
                "cwd": (ROOT / "panel").resolve(),
                "env": {},
            }
        ]
    if check_name == "jang_runtime_provenance" and python.is_file():
        configured = os.environ.get("VMLX_JANG_TOOLS_SOURCE")
        if not configured:
            return None
        jang_root = Path(configured).expanduser().resolve()
        output_dir = private_root / f"{run_id}-jang-dist"
        python_path = os.pathsep.join((str(jang_root), str(ROOT.resolve())))
        import_script = (
            "import json, pathlib, jang_tools, vmlx_engine;"
            "print('VMLINUX_IMPORT_JSON=' + json.dumps({"
            "'jang_tools':str(pathlib.Path(jang_tools.__file__).resolve()),"
            "'vmlx_engine':str(pathlib.Path(vmlx_engine.__file__).resolve())"
            "},sort_keys=True))"
        )
        return [
            {
                "command_id": "jang_build",
                "argv": [
                    str(python),
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(output_dir),
                ],
                "cwd": jang_root,
                "env": {},
            },
            {
                "command_id": "jang_import",
                "argv": [str(python), "-c", import_script],
                "cwd": ROOT.resolve(),
                "env": {"PYTHONPATH": python_path},
            },
            {
                "command_id": "jang_test",
                "argv": [
                    str(python),
                    "-m",
                    "pytest",
                    "-s",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_laguna_loader.py",
                    "tests/test_jang_affine_storage.py",
                    "tests/test_jang_loader.py",
                ],
                "cwd": ROOT.resolve(),
                "env": {"PYTHONPATH": python_path},
            },
        ]
    return None


def _run_owned_process(
    check_name: str,
    spec: dict[str, Any],
    run_context: dict[str, Any],
) -> dict[str, Any]:
    argv = [str(value) for value in spec["argv"]]
    cwd = Path(spec["cwd"]).resolve()
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in spec.get("env", {}).items()})
    started_at = _iso_now()
    process = subprocess.Popen(  # noqa: S603 - argv comes from static allowlist
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    ended_at = _iso_now()
    return {
        "schema": OWNED_EXECUTION_SCHEMA,
        "run_id": run_context["run_id"],
        "check": check_name,
        "command_id": spec["command_id"],
        "pid": process.pid,
        "argv": argv,
        "cwd": str(cwd),
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "__stdout_bytes": stdout,
        "__stderr_bytes": stderr,
    }


def _owned_execution_facts(
    check_name: str,
    executions: list[dict[str, Any]],
    jang_state: dict[str, Any],
) -> set[str]:
    if (
        not executions
        or any(
            row.get("schema") != OWNED_EXECUTION_SCHEMA
            or not isinstance(row.get("pid"), int)
            or row.get("pid") <= 0
            or row.get("exit_code") != 0
            for row in executions
        )
    ):
        return set()
    stdout = b"\n".join(
        row.get("__stdout_bytes", b"")
        for row in executions
        if isinstance(row.get("__stdout_bytes"), bytes)
    )
    stderr = b"\n".join(
        row.get("__stderr_bytes", b"")
        for row in executions
        if isinstance(row.get("__stderr_bytes"), bytes)
    )
    try:
        terminal_text = (stdout + b"\n" + stderr).decode("utf-8")
    except UnicodeDecodeError:
        return set()
    if check_name == "full_python_suite":
        collected = re.search(r"collected\s+(\d+)\s+items?", terminal_text)
        passed = re.search(r"(?:^|[= ])(\d+)\s+passed\b", terminal_text, re.MULTILINE)
        if collected is None or passed is None:
            return set()
        baseline = OWNED_SUITE_BASELINES[check_name]
        if (
            int(collected.group(1)) < baseline["collected"]
            or int(passed.group(1)) < baseline["passed"]
            or int(collected.group(1)) != int(passed.group(1))
            or re.search(r"\b(?:failed|error|deselected)\b", terminal_text)
        ):
            return set()
        return {
            "complete_suite",
            "focused_suite_not_substituted",
            "terminal_summary_passed",
        }
    if check_name == "full_panel_suite":
        tests = re.search(r"\bTests\s+(\d+)\s+passed\b", terminal_text)
        files = re.search(r"\bTest Files\s+(\d+)\s+passed\b", terminal_text)
        baseline = OWNED_SUITE_BASELINES[check_name]
        if (
            tests is None
            or files is None
            or int(tests.group(1)) < baseline["passed"]
            or int(files.group(1)) < baseline["files"]
            or re.search(r"\bfailed\b", terminal_text)
        ):
            return set()
        return {
            "complete_suite",
            "focused_suite_not_substituted",
            "terminal_summary_passed",
        }
    if check_name == "typecheck":
        return (
            set()
            if re.search(r"\berror\s+TS\d+\b", terminal_text)
            else {"terminal_summary_passed"}
        )
    if check_name == "production_build":
        required = (
            "build the electron main process successfully",
            "build the electron preload files successfully",
            "build the renderer process successfully",
        )
        return (
            {"exact_checkout_built", "terminal_summary_passed"}
            if all(value in terminal_text for value in required)
            else set()
        )
    if check_name == "jang_runtime_provenance":
        if {row.get("command_id") for row in executions} != {
            "jang_build",
            "jang_import",
            "jang_test",
        }:
            return set()
        match = re.search(r"^VMLINUX_IMPORT_JSON=(\{.*\})$", terminal_text, re.MULTILINE)
        if match is None:
            return set()
        try:
            imported = json.loads(match.group(1))
        except json.JSONDecodeError:
            return set()
        configured = os.environ.get("VMLX_JANG_TOOLS_SOURCE")
        if (
            not configured
            or not isinstance(imported, dict)
            or not path_within(
                Path(str(imported.get("jang_tools") or "")).resolve(),
                Path(configured).expanduser().resolve(),
            )
            or Path(str(imported.get("vmlx_engine") or "")).resolve()
            != (ROOT / "vmlx_engine/__init__.py").resolve()
            or jang_state.get("commit") != JANG_COMMIT
            or jang_state.get("tree") != JANG_TREE
            or jang_state.get("version") != JANG_VERSION
        ):
            return set()
        return set(REQUIRED_ASSERTIONS[check_name])
    return set()


def _execute_owned_checks(
    requested: set[str],
    private_root: Path,
    run_context: dict[str, Any],
    jang_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for check_name in sorted(requested):
        plan = _owned_command_plan(check_name, private_root, run_context["run_id"])
        if plan is None:
            results[check_name] = {"executions": [], "facts": set()}
            continue
        executions = [
            _run_owned_process(check_name, spec, run_context) for spec in plan
        ]
        results[check_name] = {
            "executions": executions,
            "facts": _owned_execution_facts(check_name, executions, jang_state),
        }
    return results


def _owned_source_facts(git_state: dict[str, Any]) -> set[str]:
    facts: set[str] = set()
    head = git_state.get("commit")
    if head and git_state.get("tree"):
        facts.add("checkout_head_exact")
    if head and git_state.get("upstream_commit") == head:
        facts.add("origin_branch_pushed")
    if head and git_state.get("remote_main_commit") == head:
        facts.add("origin_main_exact")
    # Runtime facts are deliberately absent until this preflight itself gains
    # an owned live runtime/CDP observer.
    return facts


def _observe_process(pid: int) -> dict[str, Any] | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        started = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            check=False,
        )
        command = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if (
        started.returncode != 0
        or command.returncode != 0
        or not started.stdout.strip()
        or not command.stdout.strip()
    ):
        return None
    try:
        argv = shlex.split(command.stdout.strip())
    except ValueError:
        return None
    if not argv:
        return None
    executable = Path(argv[0])
    if not executable.is_absolute() or not executable.is_file():
        return None
    return {
        "pid": pid,
        "start_identity": started.stdout.strip(),
        "argv": argv,
        "executable_path": str(executable.resolve()),
        "executable_sha256": sha256_file(executable.resolve()),
    }


def _observe_listener(host: str, port: int) -> dict[str, Any] | None:
    if host not in {"127.0.0.1", "localhost", "::1"} or not isinstance(port, int):
        return None
    try:
        completed = subprocess.run(
            [
                "/usr/sbin/lsof",
                "-nP",
                f"-iTCP@{host}:{port}",
                "-sTCP:LISTEN",
                "-Fp",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    pids = {
        int(line[1:])
        for line in completed.stdout.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }
    return (
        {"host": host, "port": port, "owner_pid": next(iter(pids))}
        if completed.returncode == 0 and len(pids) == 1
        else None
    )


def _bundle_family_contract(
    snapshot: dict[str, Any],
    health: dict[str, Any],
) -> dict[str, Any]:
    del health
    derived = snapshot["derived"]
    config = snapshot["parsed"]["config.json"]
    jang = snapshot["parsed"]["jang_config.json"]
    quantization_kind = derived["quantization_kind"]
    quantization = {
        "codec": {
            "jangtq": "turboquant_codebook",
            "jang_affine": "affine_quantized_matmul",
            "mxfp": "mxfp",
            "base_mlx": "base_mlx",
        }[quantization_kind],
        "weight_format": quantization_kind,
        "profile": nested(jang, "profile"),
        "model_types": nested(config, "model_type"),
        "architectures": nested(config, "architectures"),
        "model_family": nested(jang, "model_family"),
        "sidecar": {
            "jang_config": bool(snapshot["parsed"]["jang_config.json"]),
            "jangtq_runtime": quantization_kind == "jangtq",
        },
    }
    return {
        "model_name": str(
            config.get("_name_or_path")
            or jang.get("model_name")
            or snapshot["model_bundle_path"].rsplit("/", 1)[-1]
        ),
        "model_type": str(config.get("model_type") or ""),
        "engine_type": str(jang.get("engine_type") or config.get("engine_type") or ""),
        "model_bundle_fingerprint_sha256": snapshot["fingerprint_sha256"],
        "bundle_config_hashes": {
            name: row["sha256"] for name, row in snapshot["files"].items()
        },
        "quantization": quantization,
        "mtp": {
            "runtime_active": derived["mtp"],
            "status": (
                "native_runtime_active" if derived["mtp"] else "not_configured"
            ),
            "issues": [],
        },
        "routing": {
            "n_routed_experts": (
                max(
                    [
                        int(value)
                        for value in _nested_values(
                            snapshot["parsed"]["config.json"],
                            {
                                "num_experts",
                                "n_routed_experts",
                                "num_local_experts",
                            },
                        )
                        if isinstance(value, int) and not isinstance(value, bool)
                    ]
                    or [0]
                )
            )
        },
        "native_cache": {
            "family": derived["native_cache"],
            "cache_subtype": derived["native_cache"],
            "generic_turboquant_kv": derived["native_cache"] == "standard_kv",
        },
    }


def _common_v4_binding(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    *,
    require_electron: bool = False,
    require_gateway: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    if not _artifact_in_run_window(artifact, payload):
        return None
    source = artifact.get("source")
    context = payload.get("__validation_context")
    binding = artifact.get("binding")
    health = artifact.get("health")
    if (
        not isinstance(source, dict)
        or not isinstance(context, dict)
        or source.get("commit") != context.get("source_commit")
        or source.get("tree") != context.get("source_tree")
        or not isinstance(binding, dict)
        or not isinstance(health, dict)
    ):
        return None
    bundle_path = binding.get("bundle_path")
    snapshot = _read_bundle_directory_snapshot(bundle_path)
    if snapshot is None:
        return None
    derived = snapshot["derived"]
    if (
        binding.get("bundle_fingerprint_sha256")
        != snapshot["fingerprint_sha256"]
        or health.get("status") != "healthy"
        or health.get("model_loaded") is not True
        or health.get("model_bundle_path") != snapshot["model_bundle_path"]
        or health.get("bundle_fingerprint_sha256")
        != snapshot["fingerprint_sha256"]
        or health.get("quantization_kind") != derived["quantization_kind"]
        or health.get("mtp") is not derived["mtp"]
        or health.get("moe") is not derived["moe"]
        or health.get("native_cache") != derived["native_cache"]
    ):
        return None
    backend_claim = binding.get("backend_process")
    listener_claim = binding.get("direct_listener")
    if (
        not isinstance(backend_claim, dict)
        or _observe_process(backend_claim.get("pid")) != backend_claim
        or not isinstance(listener_claim, dict)
        or _observe_listener(
            str(listener_claim.get("host") or ""),
            listener_claim.get("port"),
        )
        != listener_claim
        or listener_claim.get("owner_pid") != backend_claim.get("pid")
    ):
        return None
    if require_gateway:
        gateway_claim = binding.get("gateway_process")
        gateway_listener = binding.get("gateway_listener")
        if (
            not isinstance(gateway_claim, dict)
            or _observe_process(gateway_claim.get("pid")) != gateway_claim
            or not isinstance(gateway_listener, dict)
            or _observe_listener(
                str(gateway_listener.get("host") or ""),
                gateway_listener.get("port"),
            )
            != gateway_listener
            or gateway_listener.get("owner_pid") != gateway_claim.get("pid")
        ):
            return None
    if require_electron:
        electron_claim = binding.get("electron_process")
        renderer = binding.get("renderer")
        cdp = artifact.get("cdp")
        if (
            not isinstance(electron_claim, dict)
            or _observe_process(electron_claim.get("pid")) != electron_claim
            or not isinstance(renderer, dict)
            or not SHA256_RE.fullmatch(
                str(renderer.get("build_manifest_sha256") or "")
            )
            or not isinstance(renderer.get("url"), str)
            or not isinstance(cdp, dict)
            or cdp.get("target_url") != renderer.get("url")
            or not str(cdp.get("websocket_url") or "").startswith("ws://")
            or cdp.get("electron_pid") != electron_claim.get("pid")
        ):
            return None
    normalized_binding = {
        "backend_pid": backend_claim["pid"],
        "runtime_source_hashes": context.get("runtime_source_hashes", {}),
        "model_bundle_fingerprint_sha256": snapshot["fingerprint_sha256"],
        "cache_topology_fingerprint_sha256": str(
            binding.get("cache_topology_fingerprint_sha256") or ""
        ),
        "python_source_file_count": context.get("python_source_file_count"),
        "python_source_read_error_count": context.get(
            "python_source_read_error_count"
        ),
    }
    return snapshot, _bundle_family_contract(snapshot, health), normalized_binding


def _decode_hashed_json(value: Any, digest: Any) -> Any:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(str(digest or "")):
        return None
    raw = value.encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != digest:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _successful_terminal(events: list[dict[str, Any]]) -> bool:
    terminal_events = [
        event
        for event in events
        if event.get("type")
        in {
            "terminal",
            "response.completed",
            "response.failed",
            "response.cancelled",
            "response.incomplete",
            "error",
        }
    ]
    return (
        len(terminal_events) == 1
        and terminal_events[0] is events[-1]
        and terminal_events[0].get("type") in {"terminal", "response.completed"}
        and terminal_events[0].get("status") == "completed"
    )


def _semantic_electron_turn(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    del artifacts
    if artifact.get("schema") != ELECTRON_RAW_SCHEMA:
        return None
    common = _common_v4_binding(artifact, payload, require_electron=True)
    if common is None:
        return None
    snapshot, family_contract, binding = common
    turns = artifact.get("turns")
    start_button = artifact.get("start_button_event")
    if (
        not isinstance(turns, list)
        or len(turns) < 3
        or not isinstance(start_button, dict)
        or start_button.get("trusted") is not True
        or start_button.get("model_session_id")
        != artifact.get("model_session_id")
    ):
        return None
    all_reasoning: list[str] = []
    all_content: list[str] = []
    saw_tool = False
    saw_result = False
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        request = _decode_hashed_json(
            turn.get("request_body_json"),
            turn.get("request_sha256"),
        )
        dom = _decode_hashed_json(
            turn.get("dom_snapshot_json"),
            turn.get("dom_snapshot_sha256"),
        )
        events = turn.get("events")
        if (
            not isinstance(request, dict)
            or not isinstance(dom, dict)
            or not isinstance(events, list)
            or not events
            or any(not isinstance(event, dict) for event in events)
            or [event.get("seq") for event in events] != list(range(len(events)))
            or not _successful_terminal(events)
        ):
            return None
        reasoning = "".join(
            str(event.get("text") or "")
            for event in events
            if event.get("type") == "reasoning_delta"
        )
        content = "".join(
            str(event.get("text") or "")
            for event in events
            if event.get("type") == "content_delta"
        )
        if (
            not content.strip()
            or dom.get("reasoning_text") != reasoning
            or dom.get("content_text") != content
            or dom.get("terminal") != "completed"
            or dom.get("rendering_ok") is not True
            or dom.get("coherent") is not True
            or not isinstance(dom.get("cache_stats"), dict)
            or not isinstance(dom.get("ttft_ms"), (int, float))
            or not isinstance(dom.get("decode_tps"), (int, float))
            or CONTROL_MARKER_RE.search(content)
        ):
            return None
        saw_tool = saw_tool or any(
            event.get("type") == "tool_call" for event in events
        )
        saw_result = saw_result or any(
            event.get("type") == "tool_result" for event in events
        )
        all_reasoning.append(reasoning)
        all_content.append(content)
    facts = {
        "real_start_button",
        "minimum_three_turns",
        "visible_content",
        "terminal_state",
        "cache_ttft_tps",
        "rendering",
        "coherence",
        "nonempty_final",
        "no_control_markup",
    }
    if sum(bool(value) for value in all_reasoning) >= 2:
        facts.add("reasoning_rail")
    if saw_tool and saw_result:
        facts.update({"tool_result_continuation", "exact_tool_arguments"})
    return {
        "command": "preflight-validated-v4-electron",
        "exit_code": 0,
        "recorded_at": artifact["recorded_at"],
        "facts": sorted(facts),
        "turn_count": len(turns),
        "model_session_id": artifact.get("model_session_id"),
        "model_id": artifact.get("model_id"),
        "binding": binding,
        "family_contract": family_contract,
        "bundle_snapshot_sha256": snapshot["fingerprint_sha256"],
    }


def _semantic_api_stream(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    del artifacts
    if artifact.get("schema") != API_RAW_SCHEMA:
        return None
    common = _common_v4_binding(artifact, payload, require_gateway=True)
    if common is None:
        return None
    snapshot, family_contract, binding = common
    flows = artifact.get("flows")
    if not isinstance(flows, list) or not flows:
        return None
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], bytes]]] = {}
    endpoints = {
        "chat": "/v1/chat/completions",
        "responses": "/v1/responses",
        "anthropic": "/v1/messages",
        "ollama": "/api/chat",
    }
    for flow in flows:
        if not isinstance(flow, dict):
            return None
        protocol = str(flow.get("protocol") or "")
        route = str(flow.get("route") or "")
        request = _decode_hashed_json(
            flow.get("request_body_json"),
            flow.get("request_sha256"),
        )
        response_text = flow.get("response_stream")
        if (
            protocol not in endpoints
            or route not in {"direct", "gateway"}
            or flow.get("endpoint") != endpoints[protocol]
            or not isinstance(request, dict)
            or not isinstance(response_text, str)
            or hashlib.sha256(response_text.encode()).hexdigest()
            != flow.get("response_sha256")
        ):
            return None
        raw_response = response_text.encode()
        parsed = _parse_raw_protocol_stream(protocol, raw_response)
        if (
            parsed is None
            or parsed["terminals"] != 1
            or parsed["terminal_last"] is not True
        ):
            return None
        grouped.setdefault((protocol, route), []).append((request, raw_response))
    protocols = sorted({key[0] for key in grouped})
    if any(
        len(grouped.get((protocol, route), [])) != 3
        for protocol in protocols
        for route in ("direct", "gateway")
    ):
        return None
    facts_by_protocol: dict[str, list[str]] = {}
    for protocol in protocols:
        route_facts = [
            _api_flow_facts_from_raw(
                protocol,
                [row[0] for row in grouped[(protocol, route)]],
                [row[1] for row in grouped[(protocol, route)]],
            )
            for route in ("direct", "gateway")
        ]
        facts_by_protocol[protocol] = sorted(set.intersection(*route_facts))
    return {
        "command": "preflight-validated-v4-api",
        "exit_code": 0,
        "recorded_at": artifact["recorded_at"],
        "facts_by_protocol": facts_by_protocol,
        "protocols": protocols,
        "turn_count": 3,
        "model_session_id": artifact.get("model_session_id"),
        "model_id": artifact.get("model_id"),
        "binding": binding,
        "family_contract": family_contract,
        "bundle_snapshot_sha256": snapshot["fingerprint_sha256"],
    }


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        matched += 1
    return matched


def _semantic_cache_observation(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    del artifacts
    if artifact.get("schema") != CACHE_RAW_SCHEMA:
        return None
    common = _common_v4_binding(artifact, payload)
    if common is None:
        return None
    snapshot, family_contract, binding = common
    source_tokens = artifact.get("source_tokens")
    candidate_tokens = artifact.get("candidate_tokens")
    telemetry = artifact.get("telemetry")
    if (
        not isinstance(source_tokens, list)
        or not isinstance(candidate_tokens, list)
        or any(not isinstance(value, int) for value in source_tokens + candidate_tokens)
        or not isinstance(telemetry, dict)
    ):
        return None
    lcp = _longest_common_prefix(source_tokens, candidate_tokens)
    suffix = len(candidate_tokens) - lcp
    events = telemetry.get("events")
    if (
        lcp <= 0
        or lcp >= len(candidate_tokens)
        or telemetry.get("matched_tokens") != lcp
        or telemetry.get("suffix_prefill_tokens") != suffix
        or not isinstance(events, list)
        or any(not isinstance(event, dict) for event in events)
        or not any(event.get("type") == "ssd_store" for event in events)
        or not any(
            event.get("type") == "ssd_restore"
            and event.get("matched_tokens") == lcp
            for event in events
        )
    ):
        return None
    facts = {
        "ssd_l2_enabled",
        "longest_prefix_partial_block_hit",
        "uncached_suffix_prefilled",
        "prefill_skip_measured",
        snapshot["derived"]["native_cache"],
    }
    if telemetry.get("paged_ram") is True:
        facts.update({"paged_ram_enabled", "ram_blocks_filled"})
    else:
        facts.add("paged_ram_disabled")
    event_types = {str(event.get("type") or "") for event in events}
    event_fact_map = {
        "cross_chat_reuse": "cross_chat_reuse",
        "cross_session_reuse": "cross_session_reuse",
        "ram_lru_evict": "oldest_unused_evicted",
        "disk_refault": "disk_refault_observed",
        "restart_restore": "restart_disk_restore",
        "disk_lru_evict": "disk_oldest_unused_evicted",
        "disk_limit_enforced": "disk_size_limit_enforced",
        "ram_limit_enforced": "ram_percentage_limit_enforced",
        "ram_oom_warning": "ram_oom_warning_checked",
        "tq_q4_default": "q4_default_when_supported",
        "tq_encode_decode": "encode_decode_live",
        "tq_off": "explicit_off_honored",
        "native_cache_exception": "unsupported_architecture_exception_honored",
    }
    facts.update(
        fact for event_type, fact in event_fact_map.items() if event_type in event_types
    )
    binding["cache_topology_fingerprint_sha256"] = _canonical_json_sha256(telemetry)
    return {
        "command": "preflight-validated-v4-cache",
        "exit_code": 0,
        "recorded_at": artifact["recorded_at"],
        "facts": sorted(facts),
        "model_session_id": artifact.get("model_session_id"),
        "model_id": artifact.get("model_id"),
        "binding": binding,
        "family_contract": family_contract,
        "bundle_snapshot_sha256": snapshot["fingerprint_sha256"],
        "native_state_derived": True,
    }


def _load_v4_raw_evidence(
    check_name: str,
    payload: dict[str, Any],
    private_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    artifacts: list[dict[str, Any]] = []
    hashes: list[str] = []
    errors: list[str] = []
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return [], [], []
    for index, item in enumerate(evidence):
        label = f"{check_name} evidence[{index}]"
        if not isinstance(item, dict) or item.get("kind") != "raw":
            errors.append(f"{label} must be one raw evidence object")
            continue
        path_value = item.get("path")
        digest = str(item.get("sha256") or "").lower()
        if (
            not isinstance(path_value, str)
            or not path_value
            or not SHA256_RE.fullmatch(digest)
        ):
            errors.append(f"{label} path or SHA-256 is invalid")
            continue
        path_failures: list[str] = []
        path = private_artifact_path(
            Path(path_value),
            private_root,
            path_failures,
            label,
        )
        if path is None:
            errors.extend(path_failures)
            continue
        raw = _read_private_evidence_once(path, private_root, path_failures, label)
        if raw is None:
            errors.extend(path_failures)
            continue
        if hashlib.sha256(raw).hexdigest() != digest:
            errors.append(f"{label} SHA-256 does not match")
            continue
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        if not isinstance(decoded, dict):
            errors.append(f"{label} is not object JSON")
            continue
        artifact = dict(decoded)
        artifact["__evidence_sha256"] = digest
        artifact["__raw_bytes"] = raw
        artifacts.append(artifact)
        hashes.append(digest)
    return artifacts, hashes, errors


def validate_attestation(
    path: Path,
    git_state: dict[str, Any],
    private_root: Path,
    failures: list[str],
    *,
    run_context: dict[str, Any] | None = None,
    owned_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    private_path = private_artifact_path(
        path,
        private_root,
        failures,
        "release attestation",
    )
    raw = (
        _read_private_evidence_once(
            private_path,
            private_root,
            failures,
            "release attestation",
        )
        if private_path is not None
        else None
    )
    try:
        decoded = json.loads(raw.decode("utf-8")) if raw is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        decoded = None
    if raw is not None and not isinstance(decoded, dict):
        failures.append("release attestation is not valid object JSON")
    data = decoded if isinstance(decoded, dict) else {}
    require(
        data.get("schema") == SCHEMA,
        failures,
        f"attestation schema={data.get('schema')!r}, expected {SCHEMA}",
    )
    require(
        data.get("scope") == SCOPE,
        failures,
        f"attestation scope={data.get('scope')!r}, expected {SCOPE}",
    )
    require(
        data.get("version") == VERSION,
        failures,
        f"attestation version={data.get('version')!r}, expected {VERSION}",
    )
    context = dict(run_context or {})
    context.setdefault("source_commit", git_state.get("commit"))
    context.setdefault("source_tree", git_state.get("tree"))
    context.setdefault("observed_at", _iso_now())
    context.setdefault("runtime_source_hashes", {})
    context.setdefault("python_source_file_count", 0)
    context.setdefault("python_source_read_error_count", 0)
    checks = data.get("checks")
    if not isinstance(checks, dict):
        failures.append("attestation checks is not an object")
        checks = {}
    sanitized: dict[str, Any] = {}
    owned = owned_results or {}
    source_facts = _owned_source_facts(git_state)
    for name in REQUIRED_CHECKS:
        payload = checks.get(name)
        if not isinstance(payload, dict):
            payload = {}
        required = set(REQUIRED_ASSERTIONS[name])
        derived: set[str] = set()
        evidence_hashes: list[str] = []
        row_errors: list[str] = []
        if name == "exact_source_provenance":
            derived = source_facts & required
        elif name in OWNED_CHECK_NAMES:
            result = owned.get(name)
            if isinstance(result, dict):
                derived = set(result.get("facts") or []) & required
        else:
            artifacts, evidence_hashes, row_errors = _load_v4_raw_evidence(
                name,
                payload,
                private_root,
            )
            semantic_payload = dict(payload)
            semantic_payload["source_commit"] = git_state.get("commit")
            semantic_payload["source_tree"] = git_state.get("tree")
            semantic_payload["__validation_context"] = context
            semantics_by_kind: dict[str, list[dict[str, Any]]] = {}
            for kind in REQUIRED_RECORD_KINDS[name]:
                semantic = _derive_semantic_record(
                    kind,
                    artifacts,
                    semantic_payload,
                )
                if semantic is not None:
                    semantics_by_kind.setdefault(kind, []).append(semantic)
            derived = _derived_assertions_for_check(name, semantics_by_kind)
        assertions = {
            assertion: assertion in derived
            for assertion in REQUIRED_ASSERTIONS[name]
        }
        if derived == required:
            status = "pass"
        elif row_errors:
            status = "fail"
        else:
            status = "blocked"
        sanitized[name] = {
            "status": status,
            "source_commit": git_state.get("commit"),
            "source_tree": git_state.get("tree"),
            "assertions": assertions,
            "evidence_sha256": evidence_hashes,
        }
        if status != "pass":
            failures.append(
                f"{name} {status}: "
                + (
                    "; ".join(row_errors)
                    if row_errors
                    else "required assertions were not independently derived"
                )
            )
    extra = sorted(set(checks) - set(REQUIRED_CHECKS))
    require(not extra, failures, f"attestation has unrecognized checks: {extra}")
    return sanitized


def _legacy_main_v4() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=VERSION)
    parser.add_argument(
        "--attestation",
        type=Path,
        default=(
            Path(os.environ["VMLX_R18_RELEASE_ATTESTATION"])
            if os.environ.get("VMLX_R18_RELEASE_ATTESTATION")
            else None
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build/current-scoped-release-preflight-18.json",
    )
    parser.add_argument(
        "--private-evidence-root",
        type=Path,
        default=(
            Path(os.environ["VMLX_R18_PRIVATE_EVIDENCE_ROOT"])
            if os.environ.get("VMLX_R18_PRIVATE_EVIDENCE_ROOT")
            else None
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit ID shared with evidence producers started for this preflight",
    )
    parser.add_argument(
        "--run-owned-check",
        action="append",
        choices=OWNED_CHECK_NAMES,
        default=[],
        help="Execute one release check as a child owned by this preflight",
    )
    args = parser.parse_args()
    failures: list[str] = []
    run_context = {
        "run_id": args.run_id or f"r18-{uuid4()}",
        "started_at": _iso_now(),
    }
    require(
        args.expected_version == VERSION,
        failures,
        f"unsupported expected version {args.expected_version!r}",
    )
    require(
        args.attestation is not None,
        failures,
        "missing --attestation or VMLX_R18_RELEASE_ATTESTATION",
    )
    versions = validate_versions(failures)
    git_state = validate_git_state(failures)
    jang_state = validate_jang_source(failures)
    private_root = validate_private_evidence_root(
        args.private_evidence_root,
        failures,
    )
    owned_results = (
        _execute_owned_checks(
            set(args.run_owned_check),
            private_root,
            run_context,
            jang_state,
        )
        if private_root is not None
        else {}
    )
    source_attestation = release_runtime_source_attestation()
    run_context.update(
        {
            "observed_at": _iso_now(),
            "source_commit": git_state.get("commit"),
            "source_tree": git_state.get("tree"),
            "runtime_source_hashes": {
                key: source_attestation.get(key)
                for key in (
                    "server_module_sha256",
                    "package_init_sha256",
                    "python_source_tree_sha256",
                )
            },
            "python_source_file_count": source_attestation.get(
                "python_source_file_count"
            ),
            "python_source_read_error_count": source_attestation.get(
                "python_source_read_error_count"
            ),
        }
    )
    checks = (
        validate_attestation(
            args.attestation,
            git_state,
            private_root,
            failures,
            run_context=run_context,
            owned_results=owned_results,
        )
        if args.attestation is not None and private_root is not None
        else {}
    )
    manifest = {
        "schema_version": 1,
        "scope": SCOPE,
        "version": VERSION,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "versions": versions,
        "source": git_state,
        "jang": jang_state,
        "run": {
            "run_id": run_context["run_id"],
            "started_at": run_context["started_at"],
            "observed_at": run_context["observed_at"],
            "owned_checks_requested": sorted(set(args.run_owned_check)),
        },
        "owned_executions": {
            check: [
                {
                    key: value
                    for key, value in execution.items()
                    if not key.startswith("__")
                }
                for execution in result.get("executions", [])
            ]
            for check, result in owned_results.items()
        },
        "checks": checks,
        "downstream_release_gates": [
            "bundle and verify Python from the attested clean JANG source",
            "build Developer-ID-signed Sequoia and Tahoe DMGs",
            "Apple notarization, stapling, Gatekeeper, and codesign validation",
            "mount both DMGs and inspect exact Electron-only contents",
            "installed-app Electron and raw API smoke on both DMG flavors",
            "tag exact source and publish PyPI, GitHub, Homebrew, feeds, and websites",
            "run the live public release-surface contract",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"scope={SCOPE}")
    print(f"version={VERSION}")
    print(f"status={manifest['status']}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if not failures else 1


def _v5_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=VERSION)
    parser.add_argument("--private-evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--native-bundle-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-model", required=True)
    parser.add_argument("--direct-base-url", required=True)
    parser.add_argument("--gateway-base-url", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--gateway-health-url", required=True)
    parser.add_argument("--cdp-url", required=True)
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--gateway-pid", type=int, required=True)
    parser.add_argument("--electron-pid", type=int, required=True)
    parser.add_argument("--jang-source", type=Path, required=True)
    return parser


def _v5_public_execution_rows(
    owned: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            {
                key: value
                for key, value in execution.items()
                if not key.startswith("__")
            }
            for execution in result.get("executions") or []
        ]
        for name, result in owned.items()
    }


def _v5_run(
    args: argparse.Namespace,
    *,
    hooks: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    hooks = hooks or {}
    failures: list[str] = []
    nonce = secrets.token_hex(16)
    run_context = {
        "run_id": args.run_id or f"r18-{uuid4()}",
        "nonce": nonce,
        "started_at": _iso_now(),
    }
    if args.expected_version != VERSION:
        failures.append(f"unsupported expected version {args.expected_version!r}")
    version_observer = hooks.get("version_observer")
    versions = (
        version_observer(failures)
        if callable(version_observer)
        else validate_versions(failures)
    )
    private_root_observer = hooks.get("private_root_observer")
    private_root = (
        private_root_observer(args.private_evidence_root, failures)
        if callable(private_root_observer)
        else validate_private_evidence_root(args.private_evidence_root, failures)
    )
    if private_root is None:
        raise RuntimeError("private evidence root is invalid")
    run_dir = _v5_make_run_directory(
        private_root,
        run_context["run_id"],
        nonce,
    )
    source_observer = hooks.get("source_observer", _v5_git_snapshot)
    before_source = source_observer()
    bundle_snapshot = _read_bundle_directory_snapshot(args.bundle_root)
    native_bundle_snapshot = _read_bundle_directory_snapshot(
        args.native_bundle_root
    )
    if (
        bundle_snapshot is None
        or native_bundle_snapshot is None
        or bundle_snapshot["fingerprint_sha256"]
        == native_bundle_snapshot["fingerprint_sha256"]
        or bundle_snapshot["model_bundle_path"]
        == native_bundle_snapshot["model_bundle_path"]
        or native_bundle_snapshot["derived"]["native_cache"]
        not in {
            "minimax_m3_sparse",
            "dsv4_composite",
            "openpangu_native",
            "cca",
        }
    ):
        raise RuntimeError("representative bundle snapshots are unsafe or incomplete")
    representatives = {
        V5_PRIMARY_REPRESENTATIVE_ID: {
            "model": args.model,
            "bundle": bundle_snapshot,
        },
        V5_NATIVE_REPRESENTATIVE_ID: {
            "model": args.native_model,
            "bundle": native_bundle_snapshot,
        },
    }
    jang_observer = hooks.get("jang_observer")
    jang_state = (
        jang_observer(failures)
        if callable(jang_observer)
        else validate_jang_source(failures)
    )
    owned_plan_provider = hooks.get("owned_check_plan_provider")
    owned_plans = (
        owned_plan_provider(run_dir, args.jang_source)
        if callable(owned_plan_provider)
        else _v5_default_owned_check_plans(run_dir, args.jang_source)
    )
    owned_checks = _v5_execute_owned_release_checks(
        owned_plans,
        run_context,
        run_dir,
        jang_state,
    )
    producer_plan_provider = hooks.get("producer_plan_provider")
    producer_plans = (
        producer_plan_provider(args, run_dir)
        if callable(producer_plan_provider)
        else _v5_default_producer_plans(args)
    )
    configured_hold_observer = hooks.get("hold_observer")

    def observe_held_runtime(binding: dict[str, Any]) -> dict[str, Any]:
        if callable(configured_hold_observer):
            return configured_hold_observer(binding)
        representative = representatives.get(binding.get("representative_id"))
        held_bundle = (
            representative.get("bundle")
            if isinstance(representative, dict)
            else None
        )
        if not isinstance(held_bundle, dict):
            raise RuntimeError("held runtime names an unknown representative")
        held_args = argparse.Namespace(
            **{
                **vars(args),
                "direct_base_url": binding["direct_base_url"],
                "gateway_base_url": binding["gateway_base_url"],
                "health_url": binding["health_url"],
                "cdp_url": binding["cdp_url"],
                "backend_pid": binding["backend_pid"],
                "gateway_pid": binding["gateway_pid"],
                "electron_pid": binding["electron_pid"],
            }
        )
        return _v5_independent_runtime_observation(
            held_args,
            before_source,
            held_bundle,
            hooks,
        )

    common_expected = {
        "source_commit": before_source["commit"],
        "source_tree": before_source["tree"],
        "direct_base_url": args.direct_base_url,
        "gateway_base_url": args.gateway_base_url,
        "health_url": args.health_url,
        "direct_health_url": args.health_url,
        "gateway_health_url": args.gateway_health_url,
        "cdp_url": args.cdp_url,
        "electron_pid": args.electron_pid,
        "gateway_pid": args.gateway_pid,
    }
    producer_results = _v5_execute_producers(
        producer_plans,
        run_context,
        run_dir,
        expected_binding={
            representative_id: {
                **common_expected,
                "model": representative["model"],
                "model_bundle_path": representative["bundle"][
                    "model_bundle_path"
                ],
                "bundle_fingerprint_sha256": representative["bundle"][
                    "fingerprint_sha256"
                ],
                "native_cache_policy": representative["bundle"]["derived"][
                    "native_cache"
                ],
            }
            for representative_id, representative in representatives.items()
        },
        hold_observer=observe_held_runtime,
    )
    captures = _v5_collect_owned_captures(producer_results, run_context)
    runtime = producer_results["ui"].get("hold_observation")
    if not isinstance(runtime, dict) or not {
        "health_bytes_sha256",
        "dom_bytes_sha256",
        "backend",
        "gateway",
        "electron",
        "runtime_hashes",
    } <= set(runtime):
        raise RuntimeError("held UI runtime observation is incomplete")
    # This second observation is deliberately after every owned child has
    # completed.  Source drift or a newly dirty checkout invalidates all rows.
    after_source = source_observer()
    checks, check_failures = _v5_validate_release_checks(
        before_source=before_source,
        after_source=after_source,
        runtime=runtime,
        representatives=representatives,
        owned_checks=owned_checks,
        captures=captures,
    )
    failures.extend(check_failures)
    run_context["completed_at"] = _iso_now()
    manifest: dict[str, Any] = {
        "schema": V5_MANIFEST_SCHEMA,
        "scope": SCOPE,
        "version": VERSION,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "run": run_context,
        "source": {
            "commit": after_source.get("commit"),
            "tree": after_source.get("tree"),
            "upstream_commit": after_source.get("upstream_commit"),
            "remote_main_commit": after_source.get("remote_main_commit"),
            "remote_identity": after_source.get("remote_identity"),
            "main_only": after_source.get("main_only"),
            "branch_only": after_source.get("branch_only"),
            "release_diff_sha256": after_source.get("release_diff_sha256"),
            "before_sha256": _canonical_json_sha256(
                {
                    key: (
                        hashlib.sha256(value).hexdigest()
                        if isinstance(value, bytes)
                        else value
                    )
                    for key, value in before_source.items()
                }
            ),
            "after_sha256": _canonical_json_sha256(
                {
                    key: (
                        hashlib.sha256(value).hexdigest()
                        if isinstance(value, bytes)
                        else value
                    )
                    for key, value in after_source.items()
                }
            ),
        },
        "versions": versions,
        "jang": {
            key: jang_state.get(key)
            for key in (
                "version",
                "commit",
                "tree",
                "upstream_commit",
                "remote_main_commit",
                "remote_identity",
            )
        },
        "bundle": {
            "path": bundle_snapshot["model_bundle_path"],
            "fingerprint_sha256": bundle_snapshot["fingerprint_sha256"],
            "derived": bundle_snapshot["derived"],
        },
        "representatives": {
            representative_id: {
                "model": representative["model"],
                "path": representative["bundle"]["model_bundle_path"],
                "fingerprint_sha256": representative["bundle"][
                    "fingerprint_sha256"
                ],
                "derived": representative["bundle"]["derived"],
            }
            for representative_id, representative in representatives.items()
        },
        "runtime": {
            key: value
            for key, value in runtime.items()
            if key
            in {
                "health_bytes_sha256",
                "dom_bytes_sha256",
                "backend",
                "gateway",
                "electron",
                "direct_listener",
                "gateway_listener",
                "runtime_hashes",
                "bundle_fingerprint_sha256",
            }
        },
        "owned_executions": _v5_public_execution_rows(owned_checks),
        "producer_executions": {
            name: {
                key: value
                for key, value in result.items()
                if key not in {"capture"} and not key.startswith("__")
            }
            for name, result in producer_results.items()
        },
        "checks": checks,
        "followup_scope": [
            {
                "check": name,
                "status": "not_in_checkpoint_release_scope",
                "reason": (
                    "requires separate per-family or multi-architecture live "
                    "campaign; cannot be waived into a checkpoint pass"
                ),
            }
            for name in V5_FOLLOWUP_CHECKS
        ],
        "completion": {
            "state": "complete",
            "completed_at": run_context["completed_at"],
        },
    }
    manifest["completion"]["run_digest"] = _v5_manifest_digest(manifest)
    _v5_atomic_write_manifest(args.out, manifest, nonce)
    return (0 if manifest["status"] == "pass" else 1), manifest


def _v5_consume_manifest_main(argv: list[str] | None = None) -> int:
    """Revalidate one completed V5 manifest for the packaging driver.

    The live V5 preflight owns model/UI/API execution. Packaging must consume
    that immutable result instead of trying to rerun the live preflight with
    the retired V4 ``--attestation`` interface.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=VERSION)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    failures: list[str] = []
    if args.expected_version != VERSION:
        failures.append(f"unsupported expected version {args.expected_version!r}")
    versions = validate_versions(failures)
    private_root = validate_private_evidence_root(
        args.private_evidence_root,
        failures,
    )
    if private_root is None:
        print("V5 release manifest consumption failed: invalid private root")
        return 1

    manifest_path = args.manifest.expanduser().resolve()
    try:
        manifest_path.relative_to(private_root)
    except ValueError:
        failures.append("V5 release manifest is outside the private evidence root")

    raw = _read_regular_file_once(manifest_path, private_root)
    try:
        candidate = json.loads(raw) if raw is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        candidate = None
    run_id = nested(candidate, "run", "run_id") if isinstance(candidate, dict) else None
    if not isinstance(run_id, str) or not run_id:
        failures.append("V5 release manifest has no valid run ID")

    source = _v5_git_snapshot()
    jang = validate_jang_source(failures)
    manifest: dict[str, Any] | None = None
    if not failures:
        try:
            manifest = consume_v5_release_manifest(
                manifest_path,
                expected_run_id=run_id,
                expected_commit=str(source["commit"]),
                expected_tree=str(source["tree"]),
            )
        except ValueError as exc:
            failures.append(str(exc))

    source_fields = (
        "commit",
        "tree",
        "upstream_commit",
        "remote_main_commit",
        "remote_identity",
        "main_only",
        "branch_only",
        "release_diff_sha256",
    )
    jang_fields = (
        "version",
        "commit",
        "tree",
        "upstream_commit",
        "remote_main_commit",
        "remote_identity",
    )
    if manifest is not None:
        if manifest.get("version") != VERSION or manifest.get("versions") != versions:
            failures.append("V5 release manifest version state is stale")
        if any(
            nested(manifest, "source", field) != source.get(field)
            for field in source_fields
        ):
            failures.append("V5 release manifest source state is stale")
        if any(
            nested(manifest, "jang", field) != jang.get(field)
            for field in jang_fields
        ):
            failures.append("V5 release manifest JANG state is stale")
    if failures or manifest is None:
        print("V5 release manifest consumption failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    _v5_atomic_write_manifest(args.out, manifest, secrets.token_hex(16))
    print(args.out)
    print(f"scope={SCOPE}")
    print(f"version={VERSION}")
    print(f"run_id={run_id}")
    print("status=pass")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    _test_hooks: dict[str, Any] | None = None,
) -> int:
    args = _v5_parser().parse_args(argv)
    try:
        status, manifest = _v5_run(args, hooks=_test_hooks)
    except Exception as exc:  # noqa: BLE001 - release gate fails closed
        if (_test_hooks or {}).get("raise_exceptions") is True:
            raise
        message = str(exc).strip()
        detail = f": {message}" if message else ""
        print(
            "v5 preflight failed before manifest completion: "
            f"{type(exc).__name__}{detail}"
        )
        return 1
    print(args.out)
    print(f"scope={SCOPE}")
    print(f"version={VERSION}")
    print(f"run_id={manifest['run']['run_id']}")
    print(f"status={manifest['status']}")
    return status


if __name__ == "__main__":
    if "--v5-owned-worker" in sys.argv[1:]:
        raise SystemExit(_v5_owned_worker_main(sys.argv[1:]))
    if sys.argv[1:2] == ["--consume-v5-manifest"]:
        raise SystemExit(_v5_consume_manifest_main(sys.argv[2:]))
    raise SystemExit(main())
