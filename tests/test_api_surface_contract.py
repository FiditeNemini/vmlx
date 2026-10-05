from pathlib import Path
import hashlib
import json
import re

import pytest


def test_api_surface_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_api_surface_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json"
    )
    assert gate.NESTED_OUT == Path(
        "build/current-api-cache-contract-api-surface-check-20260602-cache-detail-zero-cached.json"
    )


def test_api_surface_contract_pins_named_public_surface_edges():
    from tests.cross_matrix import run_api_surface_contract as gate

    nested = gate.REQUIRED_NESTED_API_CHECKS
    panel = gate.REQUIRED_PANEL_API_TEST_MARKERS

    assert "openai_chat_sampling_kwargs" in nested
    assert "responses_sampling_kwargs" in nested
    assert "legacy_completions_output_caps_override_server_default" in nested
    assert "request_output_caps_override_server_default" in nested
    assert "prompt_context_caps_stay_separate_from_output_caps" in nested
    assert "anthropic_bundle_defaults" in nested
    assert "ollama_adapter_surface" in nested
    assert "streaming_cache_detail_usage" in nested
    assert "responses_previous_response_history" in nested
    assert "cache_reuse_endpoints" in nested
    assert "cache_stats_reuse_skip_telemetry" in nested
    assert "plain_attention_kv_status" in nested
    assert "dsv4_native_cache_status" in nested
    assert "zaya_typed_cca_status" in nested
    assert "hybrid_ssm_partial_reuse" in nested
    assert "turboquant_kv_runtime_contract" in nested
    assert "jangtq_mpp_nax_health_kernel_name" in nested
    assert "dsv4_dsml_parser_residue_rejection" in nested
    assert "turboquant_disk_roundtrip" in nested
    assert "no_generic_tq_on_hybrid_ssm" in nested

    contract_source = Path("tests/cross_matrix/run_api_surface_contract.py").read_text()
    for check in (
        "dsv4_native_cache_status",
        "zaya_typed_cca_status",
        "hybrid_ssm_partial_reuse",
        "turboquant_kv_runtime_contract",
        "turboquant_disk_roundtrip",
        "no_generic_tq_on_hybrid_ssm",
    ):
        assert f'"{check}": (' in contract_source

    assert "omits sampling and token defaults when unset so the engine resolves bundle metadata" in panel
    assert "keeps per-chat maxTokens as output budget only, never prompt context" in panel
    assert "keeps Responses maxTokens as output budget only, never prompt context" in panel
    assert "preserves DSV4 Responses max_output_tokens for Max thinking" in panel
    assert "omits malformed Ollama context values instead of poisoning max_prompt_tokens" in panel
    assert "omits unset and negative sentinels while forwarding explicit neutral sampling overrides" in panel
    assert "preserves malformed Ollama num_predict for backend validation instead of silently changing budgets" in panel
    assert "applies gateway timeout handling to Ollama embeddings proxy requests" in panel
    assert "auto-switches by model id in single-model mode before preserving streaming deltas" in panel
    assert "refuses auto-switch when previous local model cannot unload before starting target" in panel
    assert "auto-switches single-model Ollama chat while emitting incremental content chunks" in panel
    assert "auto-switches single-model Ollama generate while emitting incremental response chunks" in panel
    assert "auto-switches single-model Ollama embeddings before proxying embedding data" in panel
    assert "refuses single-model Ollama routes when previous local model cannot unload" in panel
    assert "auto-switches direct OpenAI streaming by model id without mutating payload or deltas" in panel
    assert "serializes concurrent single-model switches before starting a second target" in panel
    assert "auto-switches to a standby model by waking it before direct OpenAI streaming" in panel
    assert "auto-switches Responses API streaming by model id while preserving output text deltas" in panel
    assert "auto-switches model capability requests by path model before proxying" in panel
    assert "auto-switches cache endpoints by query model before proxying cache stats" in panel
    assert "auto-switches cache entries and clear endpoints by query model before proxying cache endpoints" in panel
    assert "auto-switches cache warm by body model before proxying warm prompts" in panel
    assert "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap" in panel
    assert "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults" in panel
    assert "Auto chat maxTokens omits per-request output caps so server default can apply" in panel
    assert "guards gateway streaming writes against client disconnect EPIPE errors" in panel
    assert "aborts Ollama backend response streams when the client response closes" in panel
    assert "aborts Ollama backend response streams when the client response closes" in panel
    assert "writes each streamed gateway response chunk once and treats EPIPE as disconnect" in panel
    assert "does not write gateway response chunks after the client socket is destroyed" in panel
    assert "does not write gateway response chunks after Node marks the response closed" in panel
    assert "does not write proxied request bodies after the backend socket is destroyed" in panel
    assert "does not write proxied request bodies after Node marks the request closed" in panel
    assert "treats top-level request handler EPIPE failures as client disconnects" in panel
    assert "treats nested broken-pipe stream errors as client disconnects" in panel
    assert "guards child process stdio stream EPIPE across app-managed process lanes" in panel
    assert "guards live proof script child stdio EPIPE while collecting e2e evidence" in panel
    assert "does not end proxied requests after the backend socket is destroyed" in panel
    assert "does not end proxied requests after Node marks the request closed" in panel
    assert "does not leave raw backend request end calls unguarded after disconnect" in panel
    assert "does not leave raw chat IPC backend request finalization unguarded" in panel
    assert "normalizes cache IPC endpoint EPIPE disconnects instead of surfacing raw unexpected errors" in panel
    assert "normalizes performance health EPIPE disconnects instead of surfacing raw unexpected errors" in panel
    assert "does not log expected chat EPIPE disconnects as raw failed-message console errors" in panel
    assert "normalizes split write EPIPE chunks before raw stderr reaches the UI" in panel
    assert "routes local image server request writes through EPIPE-aware helpers" in panel
    assert "image requests disable connection reuse and normalize reset-like socket errors" in panel
    assert "tests/chat-ui.test.ts" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_gateway_streaming_disconnect_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_gateway_ollama_proxy_response_error_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_ipc_backend_request_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_performance_health_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_child_process_stdio_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_backend_stderr_split_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_gateway_single_model_auto_switch_cache_endpoints" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()


def test_api_surface_contract_requires_performance_health_epipe_normalization():
    from tests.cross_matrix import run_api_surface_contract as gate

    source = Path("panel/src/main/ipc/performance.ts").read_text(encoding="utf-8")

    assert "function isExpectedPerformanceEndpointDisconnectError" in source
    assert "wrappedDisconnects.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))" in source
    assert "nestedErrors.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))" in source
    assert "Performance health connection lost. The model server may have stopped or restarted; retry after the session is healthy." in source

    panel_command = gate.COMMANDS["panel_api_request_builders"][1]
    assert "tests/api-gateway-ollama.test.ts" in panel_command
    assert "tests/backend-stderr.test.ts" in panel_command
    assert "tests/image-generation-state.test.ts" in panel_command
    assert "tests/image-system.test.ts" in panel_command
    assert "--reporter=verbose" in panel_command


def test_api_surface_contract_hashes_cache_ipc_disconnect_guard():
    from tests.cross_matrix import run_api_surface_contract as gate

    assert "panel/src/main/ipc/cache.ts" in gate.SOURCE_HASH_FILES


def test_api_surface_contract_hashes_performance_health_disconnect_guard():
    from tests.cross_matrix import run_api_surface_contract as gate

    assert "panel/src/main/ipc/performance.ts" in gate.SOURCE_HASH_FILES


def test_api_surface_contract_status_fails_when_required_panel_markers_are_missing():
    from tests.cross_matrix import run_api_surface_contract as gate

    source = Path("tests/cross_matrix/run_api_surface_contract.py").read_text()

    assert "all_required_panel_api_markers_present" in source
    assert "not missing_panel_markers" in source


def test_api_surface_contract_source_hash_files_exist():
    from tests.cross_matrix import run_api_surface_contract as gate

    assert "panel/scripts/live-real-ui-model-proof.mjs" in gate.SOURCE_HASH_FILES
    assert "panel/scripts/live-chat-tools-reasoning-proof.mjs" in gate.SOURCE_HASH_FILES
    assert "panel/src/main/ipc/models.ts" in gate.SOURCE_HASH_FILES
    assert "panel/src/main/backend-stderr.ts" in gate.SOURCE_HASH_FILES
    assert "panel/tests/backend-stderr.test.ts" in gate.SOURCE_HASH_FILES
    assert "panel/src/main/tools/executor.ts" in gate.SOURCE_HASH_FILES

    missing = [rel for rel in gate.SOURCE_HASH_FILES if not Path(rel).exists()]

    assert missing == []


def test_noheavy_api_cache_contract_pins_named_server_rows():
    from tests.cross_matrix import run_noheavy_api_cache_contract as gate

    required = gate.REQUIRED_NOHEAVY_API_CACHE_TEST_MARKERS

    assert "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" in required
    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_prompt_context_aliases_clamp_without_rewriting_output_caps" in required
    assert "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap" in required
    assert "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap" in required
    assert "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap" in required
    assert "test_anthropic_messages_omitted_max_tokens_uses_bundle_default" in required
    assert "test_ollama_streaming_suppresses_duplicate_done_chunks" in required
    assert "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap" in required
    assert "test_chat_stream_tracks_cache_detail_alongside_cached_tokens" in required
    assert "test_chat_stream_finish_chunks_emit_cache_detail" in required
    assert "test_responses_stream_tracks_cache_detail_alongside_cached" in required
    assert "test_responses_stream_finish_emits_cache_detail" in required
    assert "test_usage_builders_preserve_cache_detail_without_cached_tokens" in required
    assert "test_chat_stream_usage_preserves_cache_detail_without_cached_tokens" in required
    assert "test_responses_stream_usage_preserves_cache_detail_without_cached_tokens" in required
    assert "test_responses_streaming_stores_history_for_previous_response_id" in required
    assert "test_responses_streaming_reasoning_only_stores_placeholder_and_marker" in required
    assert "test_chained_response_helper_emits_warning_for_reasoning_only_predecessor" in required
    assert "test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry" in required
    assert "test_cache_entries_endpoint_lists_paged_prefix_blocks" in required
    assert "test_cache_warm_endpoint_prefills_and_stores_block_cache" in required
    assert "test_clear_cache_prefix_clears_prefix_l2_without_multimodal" in required
    assert "test_clear_cache_ram_preserves_every_prefix_l2_store" in required
    assert "test_native_cache_status_reports_dsv4_separately_from_tq_kv" in required
    assert "test_native_cache_status_reports_zaya_typed_cca" in required
    assert "test_acceleration_status_reports_internal_jangtq_acceleration_when_enabled" in required
    assert "test_responses_extracts_reasoning_rail_tool_calls_before_finalize" in required
    assert "test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw" in required
    assert "test_dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail" in required

    for command in gate.COMMANDS.values():
        if "pytest" in command:
            assert "-vv" in command
        else:
            assert "vitest" in command
            assert "--reporter" in command
            assert "verbose" in command


def test_noheavy_api_cache_contract_cache_stats_telemetry_is_first_class_check(monkeypatch):
    from tests.cross_matrix import run_noheavy_api_cache_contract as gate

    missing = "test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry"
    stdout = "\n".join(
        marker for marker in gate.REQUIRED_NOHEAVY_API_CACHE_TEST_MARKERS if marker != missing
    )

    def fake_run_command(name, cmd, cwd):
        return {
            "name": name,
            "command": cmd,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "counts": {"passed": 1, "deselected": 0},
            "stdout": stdout,
            "stdout_tail": stdout.splitlines()[-40:],
        }

    monkeypatch.setattr(gate, "_run_command", fake_run_command)
    monkeypatch.setattr(gate, "source_hashes", lambda root: {})

    artifact = gate.build_artifact(Path("."))

    assert missing in artifact["missing_markers"]
    assert artifact["checks"]["cache_stats_reuse_skip_telemetry"] is False
    assert artifact["checks"]["cache_reuse_endpoints"] is False


def test_metal_headroom_guard_contract_covers_all_public_text_surfaces():
    source = (
        Path("tests") / "cross_matrix" / "run_metal_headroom_guard_contract.py"
    ).read_text(encoding="utf-8")

    for surface in (
        "chat_completions",
        "chat_completions_stream",
        "responses",
        "responses_stream",
        "anthropic_messages",
        "anthropic_messages_stream",
        "ollama_chat",
        "ollama_chat_stream",
        "ollama_generate",
        "ollama_generate_stream",
        "cli_server_main_explicit_max_tokens",
        "cli_vmlx_engine_serve_explicit_max_tokens",
    ):
        assert surface in source
    assert "requested=8192" in source
    assert "safe_cap=1" in source
    assert "projected safe Metal headroom" in source


@pytest.fixture
def retained_api_cache_fixture(tmp_path, monkeypatch):
    from tests.cross_matrix import run_api_surface_contract as gate

    (tmp_path / "owner.py").write_text("original owner\n")
    monkeypatch.setattr(gate.api_cache_gate, "SOURCE_HASH_FILES", ("owner.py",))
    command = ["/original/venv/bin/python", "-m", "pytest", "-vv", "owner.py"]
    panel_command = gate.api_cache_gate.COMMANDS["panel_gateway_contracts"]
    monkeypatch.setattr(gate.api_cache_gate, "COMMANDS", {
        "api": command, "panel_gateway_contracts": panel_command,
    })
    receipt = {
        "status": "pass",
        "created_at": "2026-10-05T13:03:42-0700",
        "checks": {name: True for name in (*gate.REQUIRED_NESTED_API_CHECKS, "all_required_named_rows_ran")},
        "missing_markers": [],
        "source_hashes": {"owner.py": gate._sha256(tmp_path / "owner.py")},
        "commands": {"api": {
            "name": "api", "command": command, "returncode": 0,
            "elapsed_sec": 2.37, "counts": {"passed": 42},
            "stdout_tail": ["42 passed"],
        }, "panel_gateway_contracts": {
            "name": "panel_gateway_contracts", "command": panel_command,
            "returncode": 0, "elapsed_sec": 0.568, "counts": {"passed": 4},
            "stdout_tail": [
                " \x1b[32m✓\x1b[39m panel/tests/api-gateway-single-model.behavior.test.ts"
                " > ApiGateway single-model mode behavior > " + title + " 2ms"
                for title in gate.RETAINED_PANEL_GATEWAY_TITLES
            ],
        }},
    }
    return gate, tmp_path, tmp_path / "retained.json", receipt


@pytest.mark.parametrize("missing_executed_marker", [False, True])
def test_api_surface_retains_original_child_without_replaying_it(
    retained_api_cache_fixture, monkeypatch, missing_executed_marker,
):
    gate, root, path, receipt = retained_api_cache_fixture
    not_retained = "auto-switches Responses API streaming by model id while preserving output text deltas"
    receipt["commands"]["panel_gateway_contracts"]["stdout_tail"].append(
        " ↓ panel/tests/api-gateway-single-model.behavior.test.ts"
        " > ApiGateway single-model mode behavior > " + not_retained
    )
    path.write_text(json.dumps(receipt))
    original_bytes = path.read_bytes()
    session = root / "panel/src/main/ipc/sessions.ts"
    session.parent.mkdir(parents=True)
    session.write_text("\n".join([
        "function isExpectedSessionLifecycleDisconnectError",
        "function formatSessionLifecycleError",
        "Server connection lost. The model server may have stopped or restarted. Try restarting the session.",
        "formatSessionLifecycleError(error)", "formatSessionLifecycleError(data.error)",
    ]))
    monkeypatch.setattr(gate, "SOURCE_HASH_FILES", ("owner.py",))
    calls = []

    def fake_run(root, name, cwd, command):
        calls.append(name)
        assert name == "panel_api_request_builders"
        pattern = command[command.index("--testNamePattern") + 1]
        for title in gate.RETAINED_PANEL_GATEWAY_TITLES:
            assert not re.search(pattern, "ApiGateway single-model mode behavior " + title)
            assert re.search(pattern, "ApiGateway single-model mode behavior " + title + " extra case")
        assert re.search(pattern, "ApiGateway single-model mode behavior " + not_retained)
        assert re.search(pattern, "ApiGateway single-model mode behavior "
                         "allows gateway startup on ports used only by stopped or remote saved sessions")
        return {"name": name, "command": command, "returncode": 0,
                "counts": {"passed": 100},
                "stdout": "\n".join(marker for marker in gate.REQUIRED_PANEL_API_TEST_MARKERS
                    if marker not in gate.RETAINED_PANEL_GATEWAY_TITLES
                    and not (missing_executed_marker and marker == not_retained))}

    monkeypatch.setattr(gate, "_run", fake_run)
    artifact = gate.build_artifact(root, path)
    assert artifact["status"] == ("fail" if missing_executed_marker else "pass")
    assert (not_retained in artifact["missing_panel_markers"]) is missing_executed_marker
    assert artifact["results"]["panel_api_request_builders"]["counts"]["passed"] == 100
    assert calls == ["panel_api_request_builders"]
    assert "server_api_surface" not in artifact["results"]
    retained = artifact["retained_api_cache"]
    assert retained["executed_in_this_run"] is False
    assert retained["created_at"] == receipt["created_at"]
    assert retained["commands"] == receipt["commands"]
    assert retained["source_hashes"] == receipt["source_hashes"]
    assert retained["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("defect", ["missing", "skipped", "duplicate"])
def test_api_surface_rejects_unproven_retained_panel_rows(retained_api_cache_fixture, monkeypatch, defect):
    gate, root, path, receipt = retained_api_cache_fixture
    tail = receipt["commands"]["panel_gateway_contracts"]["stdout_tail"]
    if defect == "missing":
        tail.pop(0)
    elif defect == "skipped":
        tail[0] = tail[0].replace("✓", "↓")
    else:
        tail.append(tail[0])
    path.write_text(json.dumps(receipt))

    def never_run(*args):
        pytest.fail("unproven retained title must fail before any child executes")

    monkeypatch.setattr(gate, "_run", never_run)
    with pytest.raises(ValueError, match="named PASS row"):
        gate.build_artifact(root, path)


@pytest.mark.parametrize("defect", [
    "absent", "invalid_json", "nonpass", "missing_check", "failed_check",
    "missing_markers", "missing_hash", "stale_source", "missing_source",
    "missing_command", "changed_command", "failed_command", "missing_time",
    "missing_command_time", "no_passed_tests",
])
def test_api_surface_rejects_invalid_retained_child_before_any_execution(
    retained_api_cache_fixture, monkeypatch, defect,
):
    gate, root, path, receipt = retained_api_cache_fixture
    if defect == "nonpass":
        receipt["status"] = "open"
    elif defect == "missing_check":
        receipt["checks"].pop(gate.REQUIRED_NESTED_API_CHECKS[0])
    elif defect == "failed_check":
        receipt["checks"]["additional_child_check"] = False
    elif defect == "missing_markers":
        receipt["missing_markers"] = ["unproven row"]
    elif defect == "missing_hash":
        receipt["source_hashes"] = {}
    elif defect == "stale_source":
        (root / "owner.py").write_text("changed owner\n")
    elif defect == "missing_source":
        (root / "owner.py").unlink()
    elif defect == "missing_command":
        receipt["commands"] = {}
    elif defect == "changed_command":
        receipt["commands"]["api"]["command"] = ["true"]
    elif defect == "failed_command":
        receipt["commands"]["api"]["returncode"] = 1
    elif defect == "missing_time":
        receipt.pop("created_at")
    elif defect == "missing_command_time":
        receipt["commands"]["api"].pop("elapsed_sec")
    elif defect == "no_passed_tests":
        receipt["commands"]["api"]["counts"]["passed"] = 0
    if defect != "absent":
        path.write_text("not json" if defect == "invalid_json" else json.dumps(receipt))

    def never_run(*args, **kwargs):
        pytest.fail("invalid retention must fail before any child execution")

    monkeypatch.setattr(gate, "_run", never_run)
    with pytest.raises((OSError, ValueError)):
        gate.build_artifact(root, path)


def test_api_surface_default_still_runs_both_children(monkeypatch):
    from tests.cross_matrix import run_api_surface_contract as gate

    calls = []

    def fake_run(root, name, cwd, command):
        calls.append(name)
        return {"returncode": 0, "counts": {"passed": 100}, "stdout": ""}

    monkeypatch.setattr(gate, "_run", fake_run)
    monkeypatch.setattr(gate, "_load_nested", lambda root: {})
    artifact = gate.build_artifact(Path("."))
    assert calls == list(gate.COMMANDS)
    assert "retained_api_cache" not in artifact
