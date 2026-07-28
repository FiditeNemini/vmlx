from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

import tests.cross_matrix.run_cache_hierarchy_live_gate as gate
from tests.cross_matrix.run_cache_hierarchy_live_gate import (
    _cache_prompts,
    _canonical_sha256,
    _counter_deltas,
    _external_artifact_dir,
    _health_attestation_snapshot,
    _health_cache_counters,
    _observe_local_listener_identity,
    _prompt_contract,
    _summarize,
    _validate_health_runtime_provenance,
    _wait_for_store_durability,
    validate_cache_rows,
    validate_probe_linkage,
)

NONCE = "cache-nonce"
MODEL = "test/model"
SOURCE = "source-sha"
CONFIG = "c" * 64
CACHE_TOPOLOGY = "d" * 64
PROMPT_CONTRACT = _prompt_contract("shared-prefix", 320)
GIT_ROOT = Path(gate.__file__).resolve().parents[2]
TEST_PYTHON = "/test/vmlx/.venv/bin/python"
TOKEN_CONTRACT = {
    "contract_version": 1,
    "method": "final-render-tokenize-no-cache",
    "surface": "responses",
    "cache_lookup_bypassed": True,
    "model_bundle_fingerprint_sha256": CONFIG,
    "cache_topology_fingerprint_sha256": CACHE_TOPOLOGY,
    "prompts": {
        label: {
            "input_sha256": hashlib.sha256(f"prompt-{label}".encode()).hexdigest(),
            "cache_prompt_token_count": 128,
            "cache_prompt_token_ids_sha256": label.lower() * 64,
            "full_cache_prompt_token_count": 128,
            "full_cache_prompt_token_ids_sha256": "e" * 64,
            "cache_key_boundary": "full_cache_prompt",
            "cache_key_boundary_removed_tokens": 0,
            "generation_prompt_suffix_tokens": 0,
            "generation_prompt_discriminator_present": True,
            "generation_prompt_discriminator_sha256": "9" * 64,
        }
        for label in ("A", "B", "C")
    },
    "longest_common_prefix_tokens": {
        "A:A": 128,
        "A:B": 127,
        "A:C": 127,
    },
}


def test_private_attestation_post_requires_and_sends_dedicated_proof_headers(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.delenv(gate.PRIVATE_ATTESTATION_TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match=gate.PRIVATE_ATTESTATION_TOKEN_ENV):
        gate._json_post(
            "http://127.0.0.1:8000/v1/cache/token-contract",
            {"contract_version": 1},
            5,
            private_attestation=True,
        )

    token = "private_" + ("q" * 48)
    monkeypatch.setenv(gate.PRIVATE_ATTESTATION_TOKEN_ENV, token)
    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    assert gate._json_post(
        "http://127.0.0.1:8000/v1/cache/token-contract",
        {"contract_version": 1},
        7,
        private_attestation=True,
    ) == {"ok": True}
    request = captured["request"]
    assert captured["timeout"] == 7
    assert request.get_header("Authorization") == f"Bearer {token}"
    assert (
        request.get_header("X-vmlx-private-proof")
        == gate.PRIVATE_ATTESTATION_PROOF_HEADER
    )
    assert token.encode() not in request.data


def _prefix_snapshot(
    *,
    disk_only: bool = False,
    resident: bool = True,
    l2_readable: bool = True,
    access_count: int = 1,
    access_time: int = 100,
) -> tuple[dict, dict]:
    resident = resident and not disk_only
    l1 = {
        "schema": "vmlx-cache-prefix-l1-snapshot-v1",
        "access_metadata_mutated": False,
        "backend_mode": "block_disk_only" if disk_only else "paged",
        "paged_ram_enabled": not disk_only,
        "disk_only": disk_only,
        "expected_blocks": 2,
        "metadata_blocks_present": 2,
        "contiguous_metadata_blocks": 2,
        "resident_payload_blocks_present": 2 if resident else 0,
        "contiguous_resident_payload_blocks": 2 if resident else 0,
        "metadata_only_blocks": 0 if resident else 2,
        "resident_payload_bytes": 2048 if resident else 0,
        "payloads_promoted_from_disk": 0,
        "terminal_metadata_present": True,
        "terminal_resident_payload_present": resident,
        "terminal_payload_from_disk": False,
    }
    readable = 2 if l2_readable else 1
    l2 = {
        "schema": "vmlx-cache-prefix-l2-snapshot-v1",
        "access_metadata_mutated": False,
        "expected_blocks": 2,
        "indexed_blocks": readable,
        "readable_blocks": readable,
        "contiguous_indexed_blocks": readable,
        "contiguous_readable_blocks": readable,
        "stale_index_blocks": 0,
        "matched_file_size_bytes": readable * 100,
        "total_access_count": readable * access_count,
        "oldest_last_accessed_ns": access_time if readable else 0,
        "newest_last_accessed_ns": access_time if readable else 0,
        "terminal_indexed": l2_readable,
        "terminal_readable": l2_readable,
        "terminal_num_tokens": 16 if l2_readable else 0,
        "terminal_last_accessed_ns": access_time if l2_readable else 0,
        "terminal_access_count": access_count if l2_readable else 0,
        "store_total_entries": 8,
        "store_total_size_bytes": 800,
        "store_max_size_bytes": 1000,
    }
    return l1, l2


def _prefix_contract(
    *,
    chain_fingerprint: str = "e" * 64,
    terminal_fingerprint: str = "f" * 64,
    token_fingerprint: str = "1" * 64,
    resident: bool = True,
    l2_readable: bool = True,
    access_count: int = 1,
    access_time: int = 100,
    disk_only: bool = False,
) -> tuple[dict, dict]:
    prompts = {
        "left": "a shared /private/example/cache final-render prefix left",
        "right": "a shared /private/example/cache final-render prefix right",
    }
    request = gate._prefix_attestation_request(
        MODEL,
        prompts,
        {"target": ("left", "right")},
    )
    l1, l2 = _prefix_snapshot(
        disk_only=disk_only,
        resident=resident,
        l2_readable=l2_readable,
        access_count=access_count,
        access_time=access_time,
    )
    contract = {
        "contract_version": 1,
        "method": gate.PREFIX_ATTESTATION_METHOD,
        "surface": "responses",
        "cache_extra_keys_contract": "generation-prompt-only-text-render-v1",
        "caller_cache_or_media_side_keys": "rejected",
        "cache_lookup_bypassed": True,
        "access_metadata_mutated": False,
        "request_sha256": _canonical_sha256(request),
        "model_bundle_fingerprint_sha256": CONFIG,
        "cache_topology_fingerprint_sha256": CACHE_TOPOLOGY,
        "block_size": 16,
        "snapshot_wall_time_ns": access_time,
        "prompts": {
            label: {
                "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "cache_prompt_token_count": 35,
                "cache_prompt_token_ids_sha256": (
                    "a" * 64 if label == "left" else "b" * 64
                ),
                "generation_prompt_suffix_tokens": 1,
                "generation_prompt_discriminator_present": True,
                "generation_prompt_discriminator_sha256": "9" * 64,
            }
            for label, prompt in prompts.items()
        },
        "prefixes": {
            "target": {
                "labels": ["left", "right"],
                "longest_common_prefix_tokens": 33,
                "reusable_prefix_tokens": 32,
                "uncached_left_tokens": 3,
                "uncached_right_tokens": 3,
                "expected_blocks": 2,
                "prefix_token_vector_sha256": token_fingerprint,
                "block_chain_fingerprint_sha256": chain_fingerprint,
                "terminal_block_fingerprint_sha256": terminal_fingerprint,
                "generation_prompt_discriminator_present": True,
                "generation_prompt_discriminator_sha256": "9" * 64,
                "l1": l1,
                "l2": l2,
            }
        },
    }
    return request, contract


def _prefix_health_attestation() -> dict:
    health_attestation, failures = _health_attestation_snapshot(_health())
    assert failures == []
    return health_attestation


def _path_free_execution() -> dict:
    return {
        "response_id": "resp-l2",
        "response_id_consistent": True,
        "status_code": 200,
        "terminal_ok": True,
        "marker_ok": True,
        "cached_tokens": 32,
        "cache_detail": {},
        "last_cache_execution": {
            "request_id": "resp-l2",
            "cache_reuse_applied": True,
            "cache_outcome": "hit",
            "cache_detail": "block-disk",
            "prompt_tokens": 35,
            "cached_tokens": 32,
            "uncached_prompt_tokens": 3,
            "prefill_tokens": 3,
            "disk_blocks": 2,
        },
    }


def _hybrid_path_free_execution() -> dict:
    execution = _path_free_execution()
    execution["health_counter_deltas"] = {
        "block_disk_cache.disk_hits": 2,
    }
    _hybridize_hit(execution, disk=True)
    execution["cache_detail"] = {
        "source": "paged+ssm+disk+tq-native",
    }
    return execution


def _l2_eviction_observation(*, disk_only: bool = False) -> dict:
    _, old_contract = _prefix_contract(
        chain_fingerprint="2" * 64,
        terminal_fingerprint="3" * 64,
        token_fingerprint="4" * 64,
        resident=not disk_only,
        access_count=1,
        access_time=100,
        disk_only=disk_only,
    )
    _, recent_contract = _prefix_contract(
        chain_fingerprint="5" * 64,
        terminal_fingerprint="6" * 64,
        token_fingerprint="7" * 64,
        resident=not disk_only,
        access_count=1,
        access_time=150,
        disk_only=disk_only,
    )
    old_before = gate._prefix_binding(old_contract, "target")
    recent_before = gate._prefix_binding(recent_contract, "target")

    recent_pre_contract = deepcopy(recent_contract)
    recent_pre_contract["snapshot_wall_time_ns"] = 200
    recent_pre_contract["prefixes"]["target"]["l1"] = _prefix_snapshot(
        disk_only=disk_only,
        resident=False,
        access_count=2,
        access_time=200,
    )[0]
    recent_pre_contract["prefixes"]["target"]["l2"] = _prefix_snapshot(
        disk_only=disk_only,
        resident=False,
        access_count=2,
        access_time=200,
    )[1]
    recent_pre = gate._prefix_binding(recent_pre_contract, "target")

    recent_post_contract = deepcopy(recent_pre_contract)
    recent_post_contract["snapshot_wall_time_ns"] = 300
    recent_post_contract["prefixes"]["target"]["l2"] = _prefix_snapshot(
        disk_only=disk_only,
        resident=True,
        access_count=3,
        access_time=300,
    )[1]
    recent_post = gate._prefix_binding(recent_post_contract, "target")

    old_final_contract = deepcopy(old_contract)
    old_final_contract["prefixes"]["target"]["l2"] = _prefix_snapshot(
        disk_only=disk_only,
        resident=False,
        l2_readable=False,
        access_count=0,
        access_time=0,
    )[1]
    old_final = gate._prefix_binding(old_final_contract, "target")
    recent_final = recent_post
    return {
        "schema": gate.L2_SIZE_EVICTION_SCHEMA,
        "scenario": "store-evict-refault",
        "source_head": SOURCE,
        "source_tree": _observed_source(SOURCE)["tree"],
        "model_bundle_fingerprint_sha256": CONFIG,
        "cache_topology_fingerprint_sha256": CACHE_TOPOLOGY,
        "saved_max_bytes": 1000,
        "peak_observed_bytes": 900,
        "final_observed_bytes": 800,
        "bounded_filler_request_count": 2,
        "old_prefix_fingerprint_sha256": "2" * 64,
        "recent_prefix_fingerprint_sha256": "5" * 64,
        "old_prefix_evicted": True,
        "recent_prefix_present": True,
        "recent_prefix_last_access_after_old": True,
        "old_before": old_before,
        "recent_before": recent_before,
        "recent_pre_refault": recent_pre,
        "recent_post_refault": recent_post,
        "evicting_filler_fence": {
            "tag": "l2_filler_001",
            "response_id": "resp-filler",
            "request_id": "resp-filler",
            "request_correlated": True,
            "ok": True,
            "post_eviction_complete": True,
            "fence_sealed": True,
            "fence_completion_generation": 2,
            "strict_physical_reconcile": True,
            "baseline_reconciliation_generation": 0,
            "global_reconciliation_generation": 1,
            "global_accounting_generation": 1,
            "global_bytes_after": 800,
            "global_max_size_bytes": 1000,
            "disk_writes_delta": 2,
            "disk_evictions_delta": 1,
            "attestation_sha256": "8" * 64,
        },
        "old_after_durable_filler": old_final,
        "recent_after_durable_filler": recent_final,
        "old_final": old_final,
        "recent_final": recent_final,
        "recent_refault_execution": _path_free_execution(),
        "write_fences": [
            {
                "tag": "filler",
                "ok": True,
                "strict_physical_reconcile": True,
                "baseline_reconciliation_generation": 0,
                "global_reconciliation_generation": 1,
                "global_accounting_generation": 1,
                "global_bytes_after": 800,
                "global_max_size_bytes": 1000,
                "attestation_sha256": "8" * 64,
            }
        ],
    }


def _l2_restart_observation(store: dict) -> dict:
    _, recent_contract = _prefix_contract(
        chain_fingerprint=store["recent_prefix_fingerprint_sha256"],
        terminal_fingerprint="6" * 64,
        token_fingerprint="7" * 64,
        resident=False,
        access_count=3,
        access_time=300,
    )
    restart_pre = gate._prefix_binding(recent_contract, "target")
    post_contract = deepcopy(recent_contract)
    post_contract["snapshot_wall_time_ns"] = 400
    post_contract["prefixes"]["target"]["l2"] = _prefix_snapshot(
        resident=True,
        access_count=4,
        access_time=400,
    )[1]
    restart_post = gate._prefix_binding(post_contract, "target")
    return {
        "schema": gate.L2_RESTART_RESTORE_SCHEMA,
        "scenario": "restart-restore",
        "source_head": SOURCE,
        "source_tree": _observed_source(SOURCE)["tree"],
        "model_bundle_fingerprint_sha256": CONFIG,
        "cache_topology_fingerprint_sha256": CACHE_TOPOLOGY,
        "restart_probe_prefix_fingerprint_sha256": store[
            "recent_prefix_fingerprint_sha256"
        ],
        "restart_restored_tokens": 32,
        "restart_disk_blocks": 2,
        "restart_uncached_tokens": 3,
        "restart_restore_source": "block-disk",
        "restart_pre": restart_pre,
        "restart_post": restart_post,
        "restart_execution": _path_free_execution(),
    }


def _validate_rows(
    phase: str,
    rows: list[dict],
    *,
    store_summary: dict | None = None,
    token_contract: dict | None = None,
    contract_profile: str = "generic",
) -> list[str]:
    return validate_cache_rows(
        phase,
        rows,
        store_summary=store_summary,
        token_contract=TOKEN_CONTRACT if token_contract is None else token_contract,
        contract_profile=contract_profile,
    )


def test_cache_hierarchy_live_gate_uses_release_semantic_artifact_schema():
    assert gate.ARTIFACT_SCHEMA == "vmlx-cache-hierarchy-live-gate-v2"


def test_cache_scenario_keeps_instructions_and_tool_schema_stable():
    prompts = {"A": "shared alpha", "B": "shared beta"}
    token_request = gate._token_contract_request(MODEL, prompts)
    prefix_request = gate._prefix_attestation_request(
        MODEL,
        prompts,
        {"target": ("A", "B")},
    )
    generated = gate._payload(MODEL, prompts["A"])

    assert token_request["request_controls"] == prefix_request["request_controls"]
    controls = token_request["request_controls"]
    assert controls["instructions"] == generated["instructions"]
    assert controls["tools"] == generated["tools"]
    assert controls["tools"][0]["name"] == "cache_contract_unused"
    assert "tool_choice" not in controls
    assert "tool_choice" not in generated
    assert generated["max_output_tokens"] == 512
    assert "cache_salt" not in controls
    assert "media_salt" not in controls


def test_prefix_attestation_sends_only_pair_referenced_prompts():
    prompts = {
        "old_store": "old store",
        "old_probe": "old probe",
        "old_restart": "unused old restart",
        "recent_store": "recent store",
        "recent_probe": "recent probe",
        "recent_restart": "unused recent restart",
    }

    request = gate._prefix_attestation_request(
        MODEL,
        prompts,
        {
            "old": ("old_store", "old_probe"),
            "recent": ("recent_store", "recent_probe"),
        },
    )

    assert request["inputs"] == {
        "old_probe": "old probe",
        "old_store": "old store",
        "recent_probe": "recent probe",
        "recent_store": "recent store",
    }
    assert "old_restart" not in request["inputs"]
    assert "recent_restart" not in request["inputs"]


def test_prefix_attestation_rejects_missing_pair_prompt():
    with pytest.raises(ValueError, match="references missing prompts"):
        gate._prefix_attestation_request(
            MODEL,
            {"store": "stored"},
            {"target": ("store", "missing")},
        )


def test_l2_eviction_producers_diverge_before_shared_prompt_and_omit_tools():
    prompts = gate._l2_identity_prompts(NONCE, 2)
    filler, _ = gate._l2_filler_prompt(NONCE, 2, 0)
    controls = gate._l2_scenario_request_controls()

    assert prompts["old_store"].startswith(
        f"CACHE-IDENTITY {NONCE}-l2-old\n"
    )
    assert prompts["recent_store"].startswith(
        f"CACHE-IDENTITY {NONCE}-l2-recent\n"
    )
    assert filler.startswith(f"CACHE-IDENTITY {NONCE}-l2-filler-000\n")
    assert controls == {
        "enable_thinking": False,
        "instructions": gate.L2_EVICTION_INSTRUCTIONS,
    }
    assert "tools" not in controls


def test_prefix_attestation_contract_is_path_free_and_supports_disk_only_l2():
    request, contract = _prefix_contract(disk_only=True, resident=False)

    failures = gate._validate_prefix_attestation_contract(
        contract,
        request_payload=request,
        health_attestation=_prefix_health_attestation(),
    )

    assert failures == []
    serialized = json.dumps(contract, sort_keys=True)
    assert "/private/example/cache" not in serialized
    assert '"token_ids":' not in serialized
    assert '"block_hash":' not in serialized
    assert contract["prefixes"]["target"]["l1"]["backend_mode"] == (
        "block_disk_only"
    )


def test_prefix_attestation_rejects_raw_tokens_paths_and_wrong_provenance():
    request, contract = _prefix_contract()
    contract["token_ids"] = [1, 2, 3]
    contract["debug_path"] = "/private/example/cache"
    contract["model_bundle_fingerprint_sha256"] = "9" * 64
    contract["cache_topology_fingerprint_sha256"] = "0" * 64

    failures = gate._validate_prefix_attestation_contract(
        contract,
        request_payload=request,
        health_attestation=_prefix_health_attestation(),
    )

    assert any("forbidden field" in failure for failure in failures)
    assert any("absolute local path leaked" in failure for failure in failures)
    assert any("model_bundle_fingerprint" in failure for failure in failures)
    assert any("cache_topology_fingerprint" in failure for failure in failures)


def test_l2_eviction_observation_requires_exact_identity_not_generic_counters():
    observation = _l2_eviction_observation()
    assert gate.validate_l2_size_eviction_observation(
        observation,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
        max_filler_requests=64,
    ) == []

    generic_only = {
        key: value
        for key, value in observation.items()
        if key
        not in {
            "old_before",
            "recent_before",
            "recent_pre_refault",
            "recent_post_refault",
            "evicting_filler_fence",
            "old_after_durable_filler",
            "recent_after_durable_filler",
            "old_final",
            "recent_final",
            "recent_refault_execution",
        }
    }
    generic_only["disk_hits"] = 9
    failures = gate.validate_l2_size_eviction_observation(
        generic_only,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
        max_filler_requests=64,
    )
    assert any("source prefix binding is missing" in failure for failure in failures)
    assert any("exact request execution is missing" in failure for failure in failures)


def test_l2_eviction_observation_accepts_truthful_ssd_only_state():
    observation = _l2_eviction_observation(disk_only=True)

    failures = gate.validate_l2_size_eviction_observation(
        observation,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
        max_filler_requests=64,
    )

    assert failures == []
    for label in (
        "recent_before",
        "recent_pre_refault",
        "recent_post_refault",
        "recent_after_durable_filler",
        "recent_final",
    ):
        l1 = observation[label]["l1"]
        assert l1["backend_mode"] == "block_disk_only"
        assert l1["paged_ram_enabled"] is False
        assert l1["resident_payload_blocks_present"] == 0


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    (
        (
            lambda row: row["evicting_filler_fence"].update(
                {"disk_evictions_delta": 0}
            ),
            "disk_evictions delta must be positive",
        ),
        (
            lambda row: row["recent_before"]["l2"].update(
                {"terminal_last_accessed_ns": 50}
            ),
            "old prefix was not strictly older",
        ),
        (
            lambda row: row["old_after_durable_filler"]["l2"].update(
                {"terminal_readable": True}
            ),
            "did not disappear after the durable",
        ),
        (
            lambda row: row["recent_after_durable_filler"]["l2"].update(
                {"store_total_size_bytes": 1001}
            ),
            "configured byte limit",
        ),
        (
            lambda row: row["evicting_filler_fence"].update(
                {"request_id": "resp-other"}
            ),
            "not bound to its Responses request_id",
        ),
    ),
)
def test_l2_eviction_observation_fails_closed_without_exact_filler_proof(
    mutate,
    expected_failure,
):
    observation = _l2_eviction_observation()
    mutate(observation)

    failures = gate.validate_l2_size_eviction_observation(
        observation,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
        max_filler_requests=64,
    )

    assert any(expected_failure in failure for failure in failures)


def test_l2_eviction_observation_rejects_swapped_stale_and_wrong_source():
    observation = _l2_eviction_observation()
    observation["source_tree"] = "wrong-tree"
    observation["recent_prefix_fingerprint_sha256"] = observation[
        "old_prefix_fingerprint_sha256"
    ]
    observation["recent_pre_refault"]["l2"]["terminal_readable"] = False

    failures = gate.validate_l2_size_eviction_observation(
        observation,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
        max_filler_requests=64,
    )

    assert any("source tree does not match" in failure for failure in failures)
    assert any("old and recent prefixes are identical" in failure for failure in failures)
    assert any("absent from L2 before refault" in failure for failure in failures)


def test_l2_restart_observation_binds_same_surviving_prefix_and_source():
    store = _l2_eviction_observation()
    restart = _l2_restart_observation(store)
    assert gate.validate_l2_restart_restore_observation(
        restart,
        store_observation=store,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
    ) == []

    swapped_store = deepcopy(store)
    swapped_store["recent_prefix_fingerprint_sha256"] = "9" * 64
    restart["source_head"] = "wrong-source"
    failures = gate.validate_l2_restart_restore_observation(
        restart,
        store_observation=swapped_store,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_prefix_health_attestation(),
    )
    assert any("source HEAD does not match" in failure for failure in failures)
    assert any("not the stored recent prefix" in failure for failure in failures)


def test_generic_l2_refault_keeps_legacy_semantics_outside_qwen_profile():
    execution = _path_free_execution()
    execution["last_cache_execution"].update(
        {
            "cached_tokens": 35,
            "uncached_prompt_tokens": 3,
            "prefill_tokens": 99,
        }
    )

    assert gate._validate_disk_refault_execution(
        execution,
        label="generic legacy refault",
        contract_profile="generic",
    ) == []


def test_hybrid_l2_restart_observation_retains_and_validates_typed_refault():
    store = _l2_eviction_observation()
    restart = _l2_restart_observation(store)
    restart["restart_execution"] = _hybrid_path_free_execution()

    failures = gate.validate_l2_restart_restore_observation(
        restart,
        store_observation=store,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_hybrid_health_attestation(),
    )

    assert failures == []
    execution = restart["restart_execution"]
    assert execution["native_cache"]["schema"] == "hybrid_ssm_v1"
    assert (
        execution["ssm_companion"]["last_prefix_lookup"]["checkpoint_tokens"]
        == execution["last_cache_execution"]["cached_tokens"]
    )


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    (
        (
            lambda row: row["ssm_companion"]["last_prefix_lookup"].update(
                {"matched": False}
            ),
            "SSM companion prefix lookup did not match",
        ),
        (
            lambda row: row["health_counter_deltas"].update(
                {"block_disk_cache.tq_native_hits": 0}
            ),
            "block_disk_cache.tq_native_hits did not increase",
        ),
        (
            lambda row: row["health_counter_deltas"].update(
                {"ssm_companion.disk.hits": 0}
            ),
            "ssm_companion.disk.hits did not increase",
        ),
        (
            lambda row: row["health_counter_deltas"].update(
                {"scheduler.hybrid_kv_without_ssm_tokens": 32}
            ),
            "hybrid_kv_without_ssm_tokens increased",
        ),
    ),
)
def test_hybrid_l2_restart_observation_fails_closed_on_artifact_defects(
    mutate,
    expected_failure,
):
    store = _l2_eviction_observation()
    restart = _l2_restart_observation(store)
    restart["restart_execution"] = _hybrid_path_free_execution()
    mutate(restart["restart_execution"])

    failures = gate.validate_l2_restart_restore_observation(
        restart,
        store_observation=store,
        expected_source_head=SOURCE,
        expected_source_tree=_observed_source(SOURCE)["tree"],
        health_attestation=_hybrid_health_attestation(),
    )

    assert any(expected_failure in failure for failure in failures)


def _runtime_provenance(*, pid: int) -> dict:
    source_tree_sha256, source_file_count, source_read_error_count = (
        gate._python_source_tree_digest(GIT_ROOT / "vmlx_engine")
    )
    return {
        "pid": pid,
        "server_module_relpath": "vmlx_engine/server.py",
        "server_module_sha256": hashlib.sha256(
            (GIT_ROOT / "vmlx_engine" / "server.py").read_bytes()
        ).hexdigest(),
        "package_init_relpath": "vmlx_engine/__init__.py",
        "package_init_sha256": hashlib.sha256(
            (GIT_ROOT / "vmlx_engine" / "__init__.py").read_bytes()
        ).hexdigest(),
        "python_source_tree_sha256": source_tree_sha256,
        "python_source_file_count": source_file_count,
        "python_source_read_error_count": source_read_error_count,
        "python_executable_fingerprint_sha256": hashlib.sha256(
            TEST_PYTHON.encode()
        ).hexdigest(),
    }


def test_python_source_tree_digest_reports_unreadable_files(tmp_path, monkeypatch):
    root = tmp_path / "vmlx_engine"
    root.mkdir()
    good = root / "good.py"
    bad = root / "bad.py"
    good.write_text("GOOD = True\n")
    bad.write_text("BAD = True\n")
    read_bytes = Path.read_bytes

    def _read_bytes(path: Path) -> bytes:
        if path == bad:
            raise PermissionError("test unreadable file")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    digest, count, read_errors = gate._python_source_tree_digest(root)

    assert len(digest) == 64
    assert count == 1
    assert read_errors == 1


def _health(*, pid: int = 1234) -> dict:
    return {
        "runtime_provenance": _runtime_provenance(pid=pid),
        "model_bundle_provenance": {
            "fingerprint_sha256": CONFIG,
            "config_file_count": 4,
        },
        "cache_topology_provenance": {
            "fingerprint_sha256": CACHE_TOPOLOGY,
            "paged_cache": True,
            "block_size": 16,
        },
        "scheduler": {
            "num_waiting": 0,
            "num_running": 0,
        },
        "cache": {},
    }


def _observed_engine(
    fingerprint: str,
    *,
    pid: int = 1234,
) -> dict:
    return {
        "method": "macos-lsof-ps",
        "host": "127.0.0.1",
        "port": 8001,
        "pid": pid,
        "started_at": "Fri Jul 24 10:00:00 2026",
        "command": f"{TEST_PYTHON} -m vmlx_engine.cli serve test/model",
        "cwd": str(GIT_ROOT),
        "python_executable": TEST_PYTHON,
        "launch_shape": "python-module-vmlx-engine-cli",
        "fingerprint_sha256": fingerprint,
    }


def _observed_source(head: str = SOURCE) -> dict:
    return {
        "git_root": str(GIT_ROOT),
        "head": head,
        "tree": "tree-" + head,
        "dirty": False,
        "status_porcelain": [],
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _hit_row(
    tag: str,
    *,
    cached_tokens: int,
    prompt_tokens: int = 128,
    cache_detail: str = "paged+tq",
    disk_blocks: int = 0,
    disk_hit_delta: int = 0,
    generation_suffix_tokens: int | None = None,
    block_size: int = 16,
) -> dict:
    uncached_tokens = prompt_tokens - cached_tokens
    response_id = f"resp-{tag}"
    prefill_tokens = uncached_tokens + (generation_suffix_tokens or 0)
    if cached_tokens == prompt_tokens and prefill_tokens == 0:
        prefill_tokens = 1
    execution = {
        "request_id": response_id,
        "prompt_tokens": prompt_tokens,
        "attempted_cached_tokens": cached_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_tokens,
        "prefill_tokens": prefill_tokens,
        "cache_detail": cache_detail,
        "cache_outcome": "hit",
        "cache_reuse_applied": True,
        "reconstructed": True,
        "reconstruction_ok": True,
        "disk_blocks": disk_blocks,
    }
    if generation_suffix_tokens is not None:
        execution["generation_prompt_suffix_tokens"] = generation_suffix_tokens
    return {
        "tag": tag,
        "status_code": 200,
        "marker_ok": True,
        "terminal_ok": True,
        "response_id": response_id,
        "response_ids": [response_id],
        "response_id_consistent": True,
        "cached_tokens": cached_tokens,
        "cache_detail": cache_detail,
        "scheduler_cache": {"block_size": block_size},
        "health_counter_deltas": {
            "block_disk_cache.disk_hits": disk_hit_delta,
            # This is retained as telemetry but is deliberately not a gate:
            # the block-disk counter is the source of truth for L2 reads.
            "scheduler_cache.disk_hits": disk_hit_delta,
        },
        "last_cache_execution": execution,
    }


def _cold_row() -> dict:
    response_id = "resp-cold_a"
    return {
        "tag": "cold_a",
        "status_code": 200,
        "marker_ok": True,
        "terminal_ok": True,
        "response_id": response_id,
        "response_ids": [response_id],
        "response_id_consistent": True,
        "cached_tokens": 0,
        "cache_detail": None,
        "scheduler_cache": {"block_size": 16},
        "last_cache_execution": {
            "request_id": response_id,
            "prompt_tokens": 128,
            "attempted_cached_tokens": 0,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 128,
            "prefill_tokens": 128,
            "cache_outcome": "miss",
            "cache_reuse_applied": False,
        },
    }


def _valid_store_rows() -> list[dict]:
    return [
        _cold_row(),
        _hit_row("warm_a", cached_tokens=127),
        _hit_row("partial_b", cached_tokens=112),
    ]


def _store_summary(
    *,
    nonce: str = NONCE,
    model: str = MODEL,
    engine: str = "engine-before-restart",
    source: str = SOURCE,
    config: str = CONFIG,
    cache_contract_ok: bool = True,
    gate_ok: bool = True,
    durability_ok: bool = True,
) -> dict:
    health_attestation, failures = _health_attestation_snapshot(_health())
    assert failures == []
    return {
        "phase": "store",
        "nonce": nonce,
        "model": model,
        "identity": {
            "observed_engine": _observed_engine(engine),
            "declared_source": source,
            "observed_source": _observed_source(source),
            "declared_config": config,
            "runtime_provenance": _runtime_provenance(pid=1234),
            "model_bundle_provenance": _health()[
                "model_bundle_provenance"
            ],
            "cache_topology_provenance": _health()[
                "cache_topology_provenance"
            ],
            "health_attestation_sha256": health_attestation[
                "combined_sha256"
            ],
        },
        "prompt_contract": PROMPT_CONTRACT,
        "tokenizer_lcp_contract": TOKEN_CONTRACT,
        "cache_contract_ok": cache_contract_ok,
        "gate_ok": gate_ok,
        "store_durability": {"ok": durability_ok},
        "requests": _valid_store_rows(),
    }


def _probe_metadata(
    *,
    nonce: str = NONCE,
    model: str = MODEL,
    engine: str = "engine-after-restart",
    source: str = SOURCE,
    config: str = CONFIG,
) -> dict:
    probe_health = _health(pid=5678)
    health_attestation, failures = _health_attestation_snapshot(probe_health)
    assert failures == []
    return {
        "phase": "probe",
        "nonce": nonce,
        "model": model,
        "identity": {
            "observed_engine": _observed_engine(engine, pid=5678),
            "declared_source": source,
            "observed_source": _observed_source(source),
            "declared_config": config,
            "runtime_provenance": _runtime_provenance(pid=5678),
            "model_bundle_provenance": probe_health[
                "model_bundle_provenance"
            ],
            "cache_topology_provenance": probe_health[
                "cache_topology_provenance"
            ],
            "health_attestation_sha256": health_attestation[
                "combined_sha256"
            ],
        },
        "prompt_contract": PROMPT_CONTRACT,
        "tokenizer_lcp_contract": TOKEN_CONTRACT,
    }


def _valid_probe_rows() -> list[dict]:
    restart_a = _hit_row(
        "restart_a",
        cached_tokens=127,
        cache_detail="memory",
    )
    restart_a["last_cache_execution"]["selection"] = "memory"
    restart_a["last_cache_execution"].pop("reconstructed")
    restart_a["last_cache_execution"].pop("reconstruction_ok")
    return [
        _hit_row(
            "restart_partial_c",
            cached_tokens=112,
            cache_detail="paged+tq+disk",
            disk_blocks=7,
            disk_hit_delta=7,
        ),
        # C promoted the common blocks. Exact A may now reuse RAM and need not
        # increment either disk counter or invoke worker reconstruction.
        restart_a,
    ]


def _hybrid_native_cache() -> dict:
    return {
        "family": "qwen3_5",
        "cache_type": "hybrid_ssm_typed",
        "schema": "hybrid_ssm_v1",
        "paged": True,
        "block_disk_l2": True,
        "components": [
            "attention_kv",
            "ssm_companion_state",
            "async_rederive",
        ],
        "attention_kv_storage_quantization": {
            "enabled": True,
            "mode": "storage_boundary",
            "codec": "turboquant_native",
            "bits": 4,
            "value_bits": 4,
            "applies_to": "attention_kv_layers_only",
            "ssm_policy": "native_companion_state",
        },
        "generic_turboquant_kv": {
            "enabled": True,
            "reason": "hybrid_attention_kv_only",
        },
    }


def _hybrid_health(*, pid: int = 1234) -> dict:
    health = _health(pid=pid)
    native_cache = _hybrid_native_cache()
    health["native_cache"] = deepcopy(native_cache)
    health["cache_topology_provenance"]["configuration"] = {
        "native_cache": deepcopy(native_cache),
    }
    health["scheduler"].update(
        {
            "hybrid_kv_without_ssm_hits": 7,
            "hybrid_kv_without_ssm_tokens": 448,
        }
    )
    health["cache"] = {
        "block_disk_cache": {
            "disk_hits": 13,
            "disk_misses": 3,
            "disk_writes": 21,
            "disk_evictions": 2,
            "tq_native_hits": 11,
            "tq_native_writes": 19,
        },
        "ssm_companion": {
            "last_prefix_lookup": {
                "request_id": "resp-health",
                "max_len": 112,
                "candidate_lengths": [112],
                "matched": True,
                "checkpoint_tokens": 112,
                "is_complete": True,
                "source": "exact_boundary_l1_or_l2",
            },
            "disk": {
                "hits": 5,
                "misses": 2,
                "stores": 17,
            },
        },
    }
    return health


def _hybridize_hit(
    row: dict,
    *,
    disk: bool,
) -> dict:
    execution = row["last_cache_execution"]
    cached_tokens = execution["cached_tokens"]
    execution.setdefault("attempted_cached_tokens", cached_tokens)
    detail = (
        "paged+ssm+disk+tq-native"
        if disk
        else "paged+ssm+tq-native"
    )
    row["cache_detail"] = detail
    execution.update(
        {
            "cache_detail": detail,
            "selection": "paged",
            "reconstructed": True,
            "reconstruction_ok": True,
            "dequantized": True,
            "dequantization_ok": True,
            "disk_hit": disk,
            "tq_native_blocks": max(1, execution.get("disk_blocks", 0)),
        }
    )
    row["native_cache"] = _hybrid_native_cache()
    row["ssm_companion"] = {
        "last_prefix_lookup": {
            "request_id": execution["request_id"],
            "max_len": cached_tokens,
            "candidate_lengths": [cached_tokens],
            "matched": True,
            "checkpoint_tokens": cached_tokens,
            "is_complete": True,
            "source": (
                "partial_boundary_disk_l2"
                if disk
                else "exact_boundary_l1_or_l2"
            ),
        },
        "disk": {
            "hits": 1 if disk else 0,
            "misses": 0,
            "stores": 0,
        },
    }
    row["health_counter_deltas"].update(
        {
            "scheduler.hybrid_kv_without_ssm_hits": 0,
            "scheduler.hybrid_kv_without_ssm_tokens": 0,
            "block_disk_cache.tq_native_hits": (
                execution.get("disk_blocks", 0) if disk else 0
            ),
            "block_disk_cache.tq_native_writes": 0,
            "ssm_companion.disk.hits": 1 if disk else 0,
            "ssm_companion.disk.misses": 0,
            "ssm_companion.disk.stores": 0,
        }
    )
    return row


def _hybrid_store_rows() -> list[dict]:
    rows = _valid_store_rows()
    _hybridize_hit(rows[2], disk=False)
    return rows


def _hybrid_probe_rows() -> list[dict]:
    rows = _valid_probe_rows()
    _hybridize_hit(rows[0], disk=True)
    return rows


def _hybrid_health_attestation() -> dict:
    health_attestation, failures = _health_attestation_snapshot(_hybrid_health())
    assert failures == []
    return health_attestation


def test_store_contract_accepts_cold_warm_and_longest_partial_prefix():
    rows = _valid_store_rows()

    failures = _validate_rows("store", rows)

    assert failures == []
    assert all(row["cache_contract_ok"] is True for row in rows)
    assert rows[2]["expected_shared_prefix_floor_tokens"] == 112
    assert rows[2]["last_cache_execution"]["prefill_tokens"] == 16


def test_hybrid_health_evidence_retains_typed_lookup_and_monotonic_counters():
    health = _hybrid_health()
    health["native_cache"]["model_path"] = "/private/models/qwen"
    health["native_cache"]["attention_kv_storage_quantization"][
        "cache_directory"
    ] = "/private/cache/tq"
    health["cache"]["ssm_companion"]["last_prefix_lookup"][
        "debug_path"
    ] = "/private/cache/ssm-checkpoint"

    evidence = gate._health_cache_contract_evidence(health)
    counters = _health_cache_counters(health)

    assert evidence["native_cache"]["schema"] == "hybrid_ssm_v1"
    assert evidence["ssm_companion"]["last_prefix_lookup"] == {
        "request_id": "resp-health",
        "max_len": 112,
        "candidate_lengths": [112],
        "matched": True,
        "checkpoint_tokens": 112,
        "is_complete": True,
        "source": "exact_boundary_l1_or_l2",
    }
    assert evidence["ssm_companion"]["disk"] == {
        "hits": 5,
        "misses": 2,
        "stores": 17,
    }
    assert counters["scheduler.hybrid_kv_without_ssm_hits"] == 7
    assert counters["scheduler.hybrid_kv_without_ssm_tokens"] == 448
    assert counters["block_disk_cache.tq_native_hits"] == 11
    assert counters["block_disk_cache.tq_native_writes"] == 19
    assert counters["ssm_companion.disk.hits"] == 5
    assert counters["ssm_companion.disk.misses"] == 2
    assert counters["ssm_companion.disk.stores"] == 17
    assert "/private/" not in json.dumps(evidence, sort_keys=True)
    assert "model_path" not in evidence["native_cache"]
    assert "cache_directory" not in evidence[
        "native_cache"
    ]["attention_kv_storage_quantization"]
    assert "debug_path" not in evidence["ssm_companion"]["last_prefix_lookup"]


def test_hybrid_path_free_execution_drops_lookup_debug_paths():
    row = _hybrid_probe_rows()[0]
    row["ssm_companion"]["last_prefix_lookup"][
        "debug_path"
    ] = "/private/cache/stale-lookup"

    execution = gate._path_free_execution(row)
    serialized = json.dumps(execution, sort_keys=True)

    assert "/private/" not in serialized
    assert "debug_path" not in execution["ssm_companion"]["last_prefix_lookup"]
    assert (
        execution["ssm_companion"]["last_prefix_lookup"]["request_id"]
        == execution["response_id"]
        == execution["last_cache_execution"]["request_id"]
    )


def test_hybrid_store_contract_accepts_only_paged_partial_prefix_with_tq_and_ssm():
    rows = _hybrid_store_rows()

    failures = _validate_rows(
        "store",
        rows,
        contract_profile="qwen_hybrid_ssm_tq4",
    )

    assert failures == []
    partial = rows[2]
    assert partial["independent_longest_common_prefix_tokens"] == 127
    assert partial["expected_shared_prefix_floor_tokens"] == 112
    assert partial["last_cache_execution"]["uncached_prompt_tokens"] == 16
    assert partial["last_cache_execution"]["prefill_tokens"] == 16
    assert (
        partial["ssm_companion"]["last_prefix_lookup"]["checkpoint_tokens"]
        == partial["last_cache_execution"]["cached_tokens"]
    )


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    (
        (
            lambda row: row["native_cache"][
                "attention_kv_storage_quantization"
            ].update({"bits": 8}),
            "does not attest attention-only TurboQuant q4",
        ),
        (
            lambda row: row["ssm_companion"].update(
                {"last_prefix_lookup": {}}
            ),
            "SSM companion prefix lookup did not match",
        ),
        (
            lambda row: row["ssm_companion"]["last_prefix_lookup"].update(
                {"checkpoint_tokens": 96}
            ),
            "does not equal accepted cached_tokens",
        ),
        (
            lambda row: row["ssm_companion"]["last_prefix_lookup"].update(
                {"max_len": 96}
            ),
            "does not equal attempted_cached_tokens",
        ),
        (
            lambda row: row["ssm_companion"]["last_prefix_lookup"].update(
                {"candidate_lengths": [96]}
            ),
            "SSM candidate lengths are not bound",
        ),
        (
            lambda row: row.update({"response_id": "resp-other"}),
            "hybrid cache evidence is not request-correlated",
        ),
        (
            lambda row: row["ssm_companion"]["last_prefix_lookup"].update(
                {"request_id": "resp-stale"}
            ),
            "SSM companion prefix lookup is not request-correlated",
        ),
        (
            lambda row: row["last_cache_execution"].update(
                {"tq_native_blocks": 0}
            ),
            "tq_native_blocks must be positive",
        ),
        (
            lambda row: row["health_counter_deltas"].update(
                {"scheduler.hybrid_kv_without_ssm_hits": 1}
            ),
            "hybrid_kv_without_ssm_hits increased",
        ),
        (
            lambda row: row["health_counter_deltas"].update(
                {"block_disk_cache.tq_native_writes": -1}
            ),
            "monotonic counter block_disk_cache.tq_native_writes has negative",
        ),
    ),
)
def test_hybrid_partial_prefix_contract_fails_closed_on_artifact_defects(
    mutate,
    expected_failure,
):
    rows = _hybrid_store_rows()
    mutate(rows[2])

    failures = _validate_rows(
        "store",
        rows,
        contract_profile="qwen_hybrid_ssm_tq4",
    )

    assert any(expected_failure in failure for failure in failures)


def test_hybrid_partial_prefix_requires_one_source_tokenized_complete_block():
    token_contract = deepcopy(TOKEN_CONTRACT)
    token_contract["longest_common_prefix_tokens"]["A:B"] = 15
    rows = _hybrid_store_rows()

    failures = _validate_rows(
        "store",
        rows,
        token_contract=token_contract,
        contract_profile="qwen_hybrid_ssm_tq4",
    )

    assert any(
        "tokenizer-derived reusable prefix floor=0 is not a meaningful cache proof"
        in failure
        for failure in failures
    )


def test_standard_scheduler_execution_count_includes_attested_generation_suffix():
    token_contract = deepcopy(TOKEN_CONTRACT)
    for prompt in token_contract["prompts"].values():
        prompt["generation_prompt_suffix_tokens"] = 5
    rows = _valid_store_rows()
    for row in rows:
        execution = row["last_cache_execution"]
        execution["prompt_tokens"] = 133
        execution["uncached_prompt_tokens"] = 133 - execution["cached_tokens"]
        execution["prefill_tokens"] = execution["uncached_prompt_tokens"]

    assert _validate_rows(
        "store",
        rows,
        token_contract=token_contract,
    ) == []
    assert rows[1]["last_cache_execution"]["prefill_tokens"] == 6
    assert rows[2]["last_cache_execution"]["prefill_tokens"] == 21
    assert rows[2]["expected_shared_prefix_floor_tokens"] == 112

    for bad_count in (132, 134):
        changed = deepcopy(rows)
        changed[1]["last_cache_execution"]["prompt_tokens"] = bad_count
        assert any(
            "independent tokenizer count=133" in failure
            for failure in _validate_rows(
                "store",
                changed,
                token_contract=token_contract,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("cache_prompt_token_count", "128", "cache-prompt token count is invalid"),
        (
            "full_cache_prompt_token_count",
            True,
            "full cache-prompt token count is invalid",
        ),
        (
            "generation_prompt_suffix_tokens",
            "not-an-int",
            "generation-prompt suffix token count is invalid",
        ),
        (
            "cache_key_boundary_removed_tokens",
            False,
            "cache-key boundary removal count is invalid",
        ),
    ],
)
def test_token_contract_counts_reject_coerced_or_boolean_integers(
    field,
    value,
    expected,
):
    prompt = deepcopy(TOKEN_CONTRACT["prompts"]["A"])
    prompt[field] = value

    _, _, _, failures = gate._token_contract_prompt_counts(
        prompt,
        tag="strict-counts",
    )

    assert any(expected in failure for failure in failures)


@pytest.mark.parametrize("boundary", [None, "bogus"])
def test_token_contract_rejects_missing_or_unknown_cache_key_boundary(boundary):
    prompt = deepcopy(TOKEN_CONTRACT["prompts"]["A"])
    if boundary is None:
        prompt.pop("cache_key_boundary")
    else:
        prompt["cache_key_boundary"] = boundary

    _, _, _, failures = gate._token_contract_prompt_counts(
        prompt,
        tag="strict-boundary",
    )

    assert any("cache-key boundary is invalid" in failure for failure in failures)


@pytest.mark.parametrize(
    ("boundary", "removed_tokens"),
    [
        ("full_cache_prompt", 1),
        ("mllm_re_feed_n_minus_one", 0),
    ],
)
def test_token_contract_boundary_requires_exact_removal_count(
    boundary,
    removed_tokens,
):
    prompt = deepcopy(TOKEN_CONTRACT["prompts"]["A"])
    prompt["cache_key_boundary"] = boundary
    prompt["cache_key_boundary_removed_tokens"] = removed_tokens

    _, _, _, failures = gate._token_contract_prompt_counts(
        prompt,
        tag="strict-boundary",
    )

    assert any("requires removed_tokens" in failure for failure in failures)


@pytest.mark.parametrize("value", ["not-an-int", True, -1])
def test_execution_suffix_rejects_malformed_mllm_domain_discriminator(value):
    prompt = deepcopy(TOKEN_CONTRACT["prompts"]["A"])
    execution = {"generation_prompt_suffix_tokens": value}

    _, failures = gate._expected_execution_prompt_tokens(
        prompt,
        execution,
        tag="strict-execution",
    )

    assert any(
        "execution generation-prompt suffix token count is invalid" in failure
        for failure in failures
    )


@pytest.mark.parametrize(("pair", "value"), [("A:A", "128"), ("A:B", True)])
def test_token_contract_rejects_coerced_or_boolean_lcp_counts(pair, value):
    prompts = {label: f"prompt-{label}" for label in ("A", "B", "C")}
    request = gate._token_contract_request(MODEL, prompts)
    contract = deepcopy(TOKEN_CONTRACT)
    contract["request_sha256"] = gate._canonical_sha256(request)
    contract["longest_common_prefix_tokens"][pair] = value
    health_attestation, health_failures = _health_attestation_snapshot(_health())
    assert health_failures == []

    failures = gate._validate_tokenizer_lcp_contract(
        contract,
        request_payload=request,
        health_attestation=health_attestation,
    )

    assert any(f"{pair} LCP count is invalid" in failure for failure in failures)


def test_minimax_m3_native_profile_accepts_stable_offset_and_block_floors():
    token_contract = deepcopy(TOKEN_CONTRACT)
    for prompt in token_contract["prompts"].values():
        prompt["cache_prompt_token_count"] = 130
        prompt["full_cache_prompt_token_count"] = 130
        prompt["generation_prompt_suffix_tokens"] = 4
    token_contract["longest_common_prefix_tokens"]["A:A"] = 130
    token_contract["longest_common_prefix_tokens"]["A:B"] = 127
    rows = _valid_store_rows()
    for row in rows:
        execution = row["last_cache_execution"]
        execution["prompt_tokens"] = 133
        execution["uncached_prompt_tokens"] = 133 - execution["cached_tokens"]
        execution["prefill_tokens"] = execution["uncached_prompt_tokens"]
        row["scheduler_cache"]["block_size"] = 16
    rows[1]["cached_tokens"] = 128
    rows[1]["last_cache_execution"].update(
        {
            "attempted_cached_tokens": 128,
            "cached_tokens": 128,
            "uncached_prompt_tokens": 5,
            "prefill_tokens": 5,
        }
    )

    failures = _validate_rows(
        "store",
        rows,
        token_contract=token_contract,
        contract_profile="minimax_m3_sparse_block",
    )

    assert failures == []
    assert all(row["native_prompt_token_offset"] == 3 for row in rows)
    assert rows[1]["expected_shared_prefix_floor_tokens"] == 128
    assert rows[2]["expected_shared_prefix_floor_tokens"] == 112

    unstable = deepcopy(rows)
    unstable[2]["last_cache_execution"]["prompt_tokens"] = 134
    assert any(
        "native sparse prompt offset is not stable" in failure
        for failure in _validate_rows(
            "store",
            unstable,
            token_contract=token_contract,
            contract_profile="minimax_m3_sparse_block",
        )
    )

    unrelated_suffix = deepcopy(token_contract)
    unrelated_suffix["prompts"]["A"]["generation_prompt_suffix_tokens"] = 0
    unrelated_suffix["prompts"]["B"]["generation_prompt_suffix_tokens"] = 0
    unrelated_suffix["prompts"]["C"]["generation_prompt_suffix_tokens"] = 4
    assert any(
        "minimum required-prompt template suffix allowance=0" in failure
        for failure in _validate_rows(
            "store",
            deepcopy(rows),
            token_contract=unrelated_suffix,
            contract_profile="minimax_m3_sparse_block",
        )
    )


def test_cache_contract_profile_requires_exact_minimax_m3_topology():
    health = _health()
    topology = {}
    health["cache_topology_provenance"]["configuration"] = topology
    topology["native_cache"] = {
        "family": "minimax_m3",
        "cache_type": "native_msa_sparse_kv",
        "schema": "minimax_m3_msa_v1",
        "generic_turboquant_kv": {"enabled": False},
    }
    topology["turboquant_kv_cache"] = {"enabled": False}
    topology["kv_cache_quantization"] = {"enabled": False}
    assert (
        gate._cache_contract_profile_from_health(health)
        == "minimax_m3_sparse_block"
    )

    for field, value in (
        ("family", "dsv4"),
        ("cache_type", "generic_kv"),
        ("schema", "other"),
    ):
        changed = deepcopy(health)
        changed["cache_topology_provenance"]["configuration"]["native_cache"][
            field
        ] = value
        assert gate._cache_contract_profile_from_health(changed) == "generic"

    for path in (
        ("native_cache", "generic_turboquant_kv"),
        ("turboquant_kv_cache",),
        ("kv_cache_quantization",),
    ):
        changed = deepcopy(health)
        node = changed["cache_topology_provenance"]["configuration"]
        if len(path) == 2:
            node = node[path[0]]
            key = path[1]
        else:
            key = path[0]
        node[key]["enabled"] = True
        assert gate._cache_contract_profile_from_health(changed) == "generic"


def test_cache_contract_profile_selects_hybrid_schema_before_field_validation():
    health = _hybrid_health()
    assert (
        gate._cache_contract_profile_from_health(health)
        == "qwen_hybrid_ssm_tq4"
    )

    malformed = deepcopy(health)
    malformed["cache_topology_provenance"]["configuration"]["native_cache"][
        "attention_kv_storage_quantization"
    ]["bits"] = 8
    assert (
        gate._cache_contract_profile_from_health(malformed)
        == "qwen_hybrid_ssm_tq4"
    )

    for family in ("lfm2", "nemotron_hybrid"):
        other = deepcopy(health)
        other["cache_topology_provenance"]["configuration"]["native_cache"][
            "family"
        ] = family
        assert gate._cache_contract_profile_from_health(other) == "generic"

    other_schema = deepcopy(health)
    other_schema["cache_topology_provenance"]["configuration"]["native_cache"][
        "schema"
    ] = "generic_kv_v1"
    assert gate._cache_contract_profile_from_health(other_schema) == "generic"


def test_standard_scheduler_prefill_does_not_require_generation_suffix_field():
    rows = _valid_store_rows()

    assert "generation_prompt_suffix_tokens" not in rows[2]["last_cache_execution"]
    assert _validate_rows("store", rows) == []


def test_mllm_prefill_includes_generation_prompt_suffix_tokens():
    token_contract = deepcopy(TOKEN_CONTRACT)
    for prompt in token_contract["prompts"].values():
        prompt["generation_prompt_suffix_tokens"] = 3
    rows = _valid_store_rows()
    rows[0]["last_cache_execution"]["generation_prompt_suffix_tokens"] = 3
    rows[0]["last_cache_execution"]["prefill_tokens"] = 131
    rows[1] = _hit_row(
        "warm_a",
        cached_tokens=127,
        generation_suffix_tokens=3,
    )
    rows[2] = _hit_row(
        "partial_b",
        cached_tokens=112,
        generation_suffix_tokens=3,
    )

    assert rows[2]["last_cache_execution"]["prefill_tokens"] == 19
    assert _validate_rows("store", rows, token_contract=token_contract) == []


def test_store_contract_accepts_direct_memory_and_prefix_reuse_without_rebuild():
    rows = _valid_store_rows()
    for row, detail in zip(rows[1:], ("memory", "prefix"), strict=True):
        row["cache_detail"] = detail
        row["last_cache_execution"]["cache_detail"] = detail
        row["last_cache_execution"]["selection"] = detail
        row["last_cache_execution"].pop("reconstructed")
        row["last_cache_execution"].pop("reconstruction_ok")

    failures = _validate_rows("store", rows)

    assert failures == []
    assert rows[1]["cache_contract_ok"] is True
    assert rows[2]["cache_contract_ok"] is True


def test_store_contract_rejects_http_output_terminal_only_rows():
    rows = [
        {"tag": tag, "status_code": 200, "marker_ok": True, "terminal_ok": True}
        for tag in ("cold_a", "warm_a", "partial_b")
    ]

    failures = _validate_rows("store", rows)

    assert failures
    assert all(
        any(
            "missing scheduler.last_cache_execution" in failure
            for failure in row["cache_contract_failures"]
        )
        for row in rows
    )


def test_main_fails_when_only_http_output_and_terminal_contracts_pass(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        gate,
        "_observe_local_listener_identity",
        lambda _base_url: _observed_engine("engine-store"),
    )
    monkeypatch.setattr(
        gate,
        "_observe_source_checkout",
        lambda: _observed_source(),
    )
    monkeypatch.setattr(gate, "_json_get", lambda _url, _timeout: _health())
    monkeypatch.setattr(
        gate,
        "_fetch_tokenizer_lcp_contract",
        lambda **_kwargs: (TOKEN_CONTRACT, []),
    )
    monkeypatch.setattr(
        gate,
        "_post_sse",
        lambda _url, _payload, _timeout: (200, "raw-sse", 0.01),
    )
    marker_outputs = iter(
        (
            "CACHE-HIERARCHY-http-only-A",
            "CACHE-HIERARCHY-http-only-A",
            "CACHE-HIERARCHY-http-only-B",
        )
    )
    monkeypatch.setattr(
        gate,
        "_summarize",
        lambda _raw, _elapsed, _status: {
            "status_code": 200,
            "elapsed_s": 0.01,
            "event_counts": {"response.completed": 1},
            "terminal_events": ["response.completed"],
            "output_text": next(marker_outputs),
            "reasoning_text": "",
            "usage": {},
            "cached_tokens": 0,
            "cache_detail": None,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cache_hierarchy_live_gate.py",
            "--model",
            MODEL,
            "--nonce",
            "http-only",
            "--source-identity",
            SOURCE,
            "--config-identity",
            CONFIG,
            "--artifact-dir",
            str(tmp_path),
            "--phase",
            "store",
        ],
    )

    exit_code = gate.main()
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert exit_code == 1
    assert summary["cache_contract_ok"] is False
    assert all(row["marker_ok"] and row["terminal_ok"] for row in summary["requests"])
    assert any(
        "missing scheduler.last_cache_execution" in failure
        for failure in summary["cache_contract_failures"]
    )
    assert (
        summary["identity"]["observed_engine"]["fingerprint_sha256"] == "engine-store"
    )
    assert (tmp_path / "cold_a.health-before.json").is_file()
    assert "health_counters_before" in summary["requests"][0]
    assert "health_counters_after" in summary["requests"][0]
    assert "health_counter_deltas" in summary["requests"][0]


def test_cache_prompts_differ_only_at_the_final_ascii_selector():
    prompts = _cache_prompts("shared prefix", NONCE)

    assert set(prompts) == {"A", "B", "C"}
    assert {prompt[:-1] for prompt in prompts.values()} == {
        f"shared prefix\nReply exactly CACHE-HIERARCHY-{NONCE}-"
    }
    assert {prompt[-1] for prompt in prompts.values()} == {"A", "B", "C"}
    assert all(ord(prompt[-1]) < 128 for prompt in prompts.values())


def test_restart_restore_uses_surviving_recent_chain_not_evicted_standard_chain():
    prompts = _cache_prompts("shared prefix", NONCE)

    assert gate._standard_cache_requests(
        "probe", "restart-restore", prompts
    ) == []
    assert [
        tag
        for tag, _prompt, _selector in gate._standard_cache_requests(
            "probe", "standard", prompts
        )
    ] == ["restart_partial_c", "restart_a"]


def _run_restart_restore_main(
    monkeypatch,
    tmp_path: Path,
    *,
    include_store_observation: bool = True,
    scenario_rows: list[dict] | None = None,
    scenario_failures: list[str] | None = None,
) -> tuple[int, dict, dict]:
    store = _store_summary()
    store["prompt_contract"] = _prompt_contract(
        gate._common_prefix(NONCE, 320),
        320,
    )
    store_observation = _l2_eviction_observation()
    if include_store_observation:
        store["l2_size_eviction_observation"] = store_observation
    store_path = tmp_path / "store-summary.json"
    store_path.write_text(json.dumps(store))
    calls = {"scenario": 0, "generic": 0}

    def unexpected_generic_request(*_args, **_kwargs):
        calls["generic"] += 1
        raise AssertionError("restart-restore must not probe evicted A/C rows")

    def fake_restart_scenario(**kwargs):
        calls["scenario"] += 1
        assert kwargs["store_observation"] == store.get(
            "l2_size_eviction_observation"
        )
        return (
            _l2_restart_observation(store_observation),
            (
                [
                    {
                        "status_code": 200,
                        "marker_ok": True,
                        "terminal_ok": True,
                    }
                ]
                if scenario_rows is None
                else scenario_rows
            ),
            [] if scenario_failures is None else scenario_failures,
            _health(pid=5678),
        )

    monkeypatch.setattr(
        gate,
        "_observe_local_listener_identity",
        lambda _base_url: _observed_engine(
            "engine-after-restart",
            pid=5678,
        ),
    )
    monkeypatch.setattr(gate, "_observe_source_checkout", _observed_source)
    monkeypatch.setattr(
        gate,
        "_json_get",
        lambda _url, _timeout: _health(pid=5678),
    )
    monkeypatch.setattr(
        gate,
        "_fetch_tokenizer_lcp_contract",
        lambda **_kwargs: (TOKEN_CONTRACT, []),
    )
    monkeypatch.setattr(gate, "_post_sse", unexpected_generic_request)
    monkeypatch.setattr(
        gate,
        "_run_restart_restore_scenario",
        fake_restart_scenario,
    )
    artifact_dir = tmp_path / "probe"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cache_hierarchy_live_gate.py",
            "--model",
            MODEL,
            "--nonce",
            NONCE,
            "--source-identity",
            SOURCE,
            "--config-identity",
            CONFIG,
            "--store-summary",
            str(store_path),
            "--artifact-dir",
            str(artifact_dir),
            "--phase",
            "probe",
            "--cache-scenario",
            "restart-restore",
        ],
    )

    exit_code = gate.main()
    return (
        exit_code,
        json.loads((artifact_dir / "summary.json").read_text()),
        calls,
    )


def test_restart_restore_main_runs_only_surviving_recent_chain(
    monkeypatch,
    tmp_path,
):
    exit_code, summary, calls = _run_restart_restore_main(
        monkeypatch,
        tmp_path,
    )

    assert exit_code == 0
    assert calls == {"scenario": 1, "generic": 0}
    assert summary["requests"] == []
    assert len(summary["scenario_requests"]) == 1
    assert summary["scenario_contract_ok"] is True
    assert summary["gate_ok"] is True


@pytest.mark.parametrize(
    ("include_store_observation", "scenario_rows", "scenario_failures"),
    [
        (False, None, None),
        (True, [], None),
        (True, None, ["invalid restart observation"]),
    ],
)
def test_restart_restore_main_fails_closed_on_missing_or_invalid_scenario(
    monkeypatch,
    tmp_path,
    include_store_observation,
    scenario_rows,
    scenario_failures,
):
    exit_code, summary, calls = _run_restart_restore_main(
        monkeypatch,
        tmp_path,
        include_store_observation=include_store_observation,
        scenario_rows=scenario_rows,
        scenario_failures=scenario_failures,
    )

    assert exit_code == 1
    assert calls == {"scenario": 1, "generic": 0}
    assert summary["gate_ok"] is False
    assert summary["scenario_contract_ok"] is False


def test_probe_contract_requires_unseen_c_then_allows_ram_exact_a():
    rows = _valid_probe_rows()
    store_summary = _store_summary()

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=store_summary,
    )

    assert failures == []
    assert rows[0]["cache_contract_ok"] is True
    assert rows[1]["cache_contract_ok"] is True
    assert rows[0]["health_counter_deltas"]["block_disk_cache.disk_hits"] == 7
    assert rows[1]["health_counter_deltas"]["block_disk_cache.disk_hits"] == 0
    assert "reconstructed" not in rows[1]["last_cache_execution"]
    assert "reconstruction_ok" not in rows[1]["last_cache_execution"]


def test_hybrid_probe_contract_requires_tq_and_ssm_disk_refault_deltas():
    rows = _hybrid_probe_rows()

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
        contract_profile="qwen_hybrid_ssm_tq4",
    )

    assert failures == []
    restart_partial = rows[0]
    execution = restart_partial["last_cache_execution"]
    assert execution["cache_detail"] == "paged+ssm+disk+tq-native"
    assert execution["selection"] == "paged"
    assert execution["disk_hit"] is True
    assert execution["disk_blocks"] == 7
    assert execution["tq_native_blocks"] == 7
    assert (
        restart_partial["health_counter_deltas"][
            "block_disk_cache.tq_native_hits"
        ]
        == 7
    )
    assert (
        restart_partial["health_counter_deltas"]["ssm_companion.disk.hits"]
        == 1
    )


@pytest.mark.parametrize(
    ("counter", "expected_failure"),
    (
        (
            "block_disk_cache.disk_hits",
            "block_disk_cache.disk_hits did not increase",
        ),
        (
            "block_disk_cache.tq_native_hits",
            "block_disk_cache.tq_native_hits did not increase",
        ),
        (
            "ssm_companion.disk.hits",
            "ssm_companion.disk.hits did not increase",
        ),
    ),
)
def test_hybrid_probe_contract_rejects_missing_restart_refault_delta(
    counter,
    expected_failure,
):
    rows = _hybrid_probe_rows()
    rows[0]["health_counter_deltas"][counter] = 0

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
        contract_profile="qwen_hybrid_ssm_tq4",
    )

    assert any(expected_failure in failure for failure in failures)


def test_hybrid_probe_contract_requires_one_tq_hit_per_reconstructed_tq_block():
    rows = _hybrid_probe_rows()
    rows[0]["health_counter_deltas"]["block_disk_cache.tq_native_hits"] = 1

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
        contract_profile="qwen_hybrid_ssm_tq4",
    )

    assert any(
        "TQ-native hit delta=1 is below reconstructed tq_native_blocks=7"
        in failure
        for failure in failures
    )


def test_probe_contract_still_requires_reconstruction_for_paged_exact_a():
    rows = _valid_probe_rows()
    row = rows[1]
    row["cache_detail"] = "paged+tq"
    row["last_cache_execution"]["cache_detail"] = "paged+tq"
    row["last_cache_execution"].pop("selection")

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any("reconstructed is not true" in failure for failure in failures)
    assert any("reconstruction_ok is not true" in failure for failure in failures)


def test_probe_contract_rejects_exact_a_before_unseen_c():
    rows = list(reversed(_valid_probe_rows()))

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any("required row order" in failure for failure in failures)


@pytest.mark.parametrize(
    ("field", "value", "expected_failure"),
    [
        ("cache_reuse_applied", False, "cache_reuse_applied is not true"),
        ("cache_outcome", "miss", "cache_outcome is not hit"),
        ("reconstructed", False, "reconstructed is not true"),
        ("reconstruction_ok", False, "reconstruction_ok is not true"),
        ("disk_blocks", 0, "disk_blocks must be positive"),
    ],
)
def test_probe_contract_rejects_false_c_execution_claims(
    field, value, expected_failure
):
    rows = _valid_probe_rows()
    rows[0]["last_cache_execution"][field] = value

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any(expected_failure in failure for failure in failures)
    assert rows[0]["cache_contract_ok"] is False


def test_responses_sse_id_is_bound_to_scheduler_execution_request_id():
    raw = "\n\n".join(
        [
            'event: response.created\ndata: {"response":{"id":"resp-live"}}',
            (
                "event: response.completed\n"
                'data: {"response":{"id":"resp-live","usage":{}}}'
            ),
            "",
        ]
    )

    summary = _summarize(raw, 0.1, 200)

    assert summary["response_id"] == "resp-live"
    assert summary["response_id_consistent"] is True
    assert summary["response_ids"] == ["resp-live"]


def test_cache_marker_accepts_only_exact_visible_text():
    marker = "CACHE-HIERARCHY-exact-A"

    assert gate._exact_cache_marker_observed(
        {"output_text": marker, "function_calls": []},
        marker,
    )
    assert not gate._exact_cache_marker_observed(
        {"output_text": f"prefix {marker}", "function_calls": []},
        marker,
    )
    assert not gate._exact_cache_marker_observed(
        {
            "output_text": marker,
            "function_calls": [
                {
                    "name": "other_tool",
                    "status": "completed",
                    "arguments": "{}",
                }
            ],
        },
        marker,
    )
    assert not gate._exact_cache_marker_observed(
        {
            "output_text": marker,
            "reasoning_text": "<think>leaked</think>",
            "function_calls": [],
        },
        marker,
    )


def test_cache_marker_accepts_one_exact_completed_schema_call():
    marker = "CACHE-HIERARCHY-exact-A"
    summary = {
        "output_text": "",
        "function_calls": [
            {
                "name": "cache_contract_unused",
                "status": "completed",
                "arguments": json.dumps({"value": marker}),
            }
        ],
    }

    assert gate._exact_cache_marker_observed(summary, marker)
    with_reasoning = deepcopy(summary)
    with_reasoning["reasoning_text"] = "private reasoning leaked"
    assert not gate._exact_cache_marker_observed(with_reasoning, marker)

    for change in (
        {"name": "other_tool"},
        {"status": "in_progress"},
        {"arguments": json.dumps({"value": marker, "extra": True})},
        {"arguments": "<value>raw-native-markup</value>"},
    ):
        changed = deepcopy(summary)
        changed["function_calls"][0].update(change)
        assert not gate._exact_cache_marker_observed(changed, marker)

    duplicated = deepcopy(summary)
    duplicated["function_calls"].append(deepcopy(summary["function_calls"][0]))
    assert not gate._exact_cache_marker_observed(duplicated, marker)


def test_sse_summary_retains_completed_function_call_for_cache_contract():
    marker = "CACHE-HIERARCHY-exact-A"
    raw = "\n\n".join(
        [
            (
                "event: response.output_item.done\n"
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "status": "completed",
                            "name": "cache_contract_unused",
                            "arguments": json.dumps({"value": marker}),
                        },
                    }
                )
            ),
            'event: response.completed\ndata: {"response":{"id":"resp-tool"}}',
            "",
        ]
    )

    summary = _summarize(raw, 0.1, 200)

    assert gate._exact_cache_marker_observed(summary, marker)
    assert summary["function_calls"] == [
        {
            "name": "cache_contract_unused",
            "arguments": json.dumps({"value": marker}),
            "status": "completed",
        }
    ]


def test_cache_contract_rejects_response_id_execution_mismatch():
    rows = _valid_store_rows()
    rows[1]["last_cache_execution"]["request_id"] = "resp-other"

    failures = _validate_rows("store", rows)

    assert any(
        "does not match execution request_id=resp-other" in item for item in failures
    )


def test_sse_summary_rejects_inconsistent_response_ids():
    raw = "\n\n".join(
        [
            'event: response.created\ndata: {"response":{"id":"resp-one"}}',
            'event: response.completed\ndata: {"response":{"id":"resp-two"}}',
            "",
        ]
    )

    summary = _summarize(raw, 0.1, 200)

    assert summary["response_id"] is None
    assert summary["response_id_consistent"] is False
    assert summary["response_ids"] == ["resp-one", "resp-two"]


def test_probe_contract_rejects_full_c_match_without_uncached_tail():
    rows = _valid_probe_rows()
    row = rows[0]
    row["cached_tokens"] = 128
    row["last_cache_execution"]["cached_tokens"] = 128
    row["last_cache_execution"]["attempted_cached_tokens"] = 128
    row["last_cache_execution"]["uncached_prompt_tokens"] = 0
    row["last_cache_execution"]["generation_prompt_suffix_tokens"] = 0
    row["last_cache_execution"]["prefill_tokens"] = 1

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any(
        "cached_tokens must be smaller than prompt_tokens" in item for item in failures
    )
    assert any("must leave an uncached tail" in item for item in failures)


def test_exact_a_allows_only_one_token_kickoff_for_true_full_hit():
    rows = _valid_probe_rows()
    row = rows[1]
    row["cached_tokens"] = 128
    row["last_cache_execution"]["cached_tokens"] = 128
    row["last_cache_execution"]["attempted_cached_tokens"] = 128
    row["last_cache_execution"]["uncached_prompt_tokens"] = 0
    row["last_cache_execution"]["generation_prompt_suffix_tokens"] = 0
    row["last_cache_execution"]["prefill_tokens"] = 1

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert failures == []


@pytest.mark.parametrize("prefill_tokens", [0, 2])
def test_exact_a_rejects_any_other_full_hit_kickoff(prefill_tokens):
    rows = _valid_probe_rows()
    row = rows[1]
    row["cached_tokens"] = 128
    row["last_cache_execution"]["cached_tokens"] = 128
    row["last_cache_execution"]["attempted_cached_tokens"] = 128
    row["last_cache_execution"]["uncached_prompt_tokens"] = 0
    row["last_cache_execution"]["generation_prompt_suffix_tokens"] = 0
    row["last_cache_execution"]["prefill_tokens"] = prefill_tokens

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any(
        "does not equal actual uncached tail plus optional generation suffix=1" in item
        for item in failures
    )


def test_probe_contract_rejects_prefill_that_is_not_actual_uncached_tail():
    rows = _valid_probe_rows()
    rows[0]["last_cache_execution"]["prefill_tokens"] += 1

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any(
        "does not equal actual uncached tail plus optional generation suffix" in item
        for item in failures
    )


def test_probe_contract_rejects_one_token_incidental_hit():
    rows = _valid_probe_rows()
    row = rows[0]
    row["cached_tokens"] = 1
    row["last_cache_execution"]["cached_tokens"] = 1
    row["last_cache_execution"]["attempted_cached_tokens"] = 1
    row["last_cache_execution"]["uncached_prompt_tokens"] = 127
    row["last_cache_execution"]["prefill_tokens"] = 127

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any("below expected shared-prefix floor 112" in item for item in failures)


def test_cache_contract_requires_actual_attempted_prefix_telemetry():
    rows = _valid_store_rows()
    rows[2]["last_cache_execution"].pop("attempted_cached_tokens")

    failures = _validate_rows("store", rows)

    assert any("attempted_cached_tokens is missing" in item for item in failures)


def test_cache_contract_rejects_shrinking_below_attempted_prefix_candidate():
    rows = _valid_store_rows()
    rows[2]["cached_tokens"] = 111
    rows[2]["last_cache_execution"]["cached_tokens"] = 111
    rows[2]["last_cache_execution"]["attempted_cached_tokens"] = 120
    rows[2]["last_cache_execution"]["uncached_prompt_tokens"] = 17
    rows[2]["last_cache_execution"]["prefill_tokens"] = 17

    failures = _validate_rows("store", rows)

    assert any("below expected shared-prefix floor 112" in item for item in failures)


def test_probe_contract_rejects_missing_linked_store_summary():
    failures = _validate_rows("probe", _valid_probe_rows())

    assert any("linked passing store summary is required" in item for item in failures)


def test_probe_contract_rejects_missing_c_disk_detail_and_block_counter_delta():
    rows = _valid_probe_rows()
    row = rows[0]
    row["cache_detail"] = "paged+tq"
    row["last_cache_execution"]["cache_detail"] = "paged+tq"
    row["health_counter_deltas"]["block_disk_cache.disk_hits"] = 0

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any("cache_detail does not identify disk" in item for item in failures)
    assert any(
        "block-disk /health disk_hits did not increase" in item for item in failures
    )
    assert not any("scheduler-cache /health" in item for item in failures)


def test_probe_contract_rejects_disk_counter_delta_smaller_than_disk_blocks():
    rows = _valid_probe_rows()
    rows[0]["health_counter_deltas"]["block_disk_cache.disk_hits"] = 2

    failures = _validate_rows(
        "probe",
        rows,
        store_summary=_store_summary(),
    )

    assert any("smaller than reconstructed disk_blocks=7" in item for item in failures)


def test_different_suffix_cannot_claim_longer_match_than_exact_reference():
    rows = _valid_store_rows()
    rows[1]["cached_tokens"] = 112
    rows[1]["last_cache_execution"]["cached_tokens"] = 112
    rows[1]["last_cache_execution"]["attempted_cached_tokens"] = 112
    rows[1]["last_cache_execution"]["uncached_prompt_tokens"] = 16
    rows[1]["last_cache_execution"]["prefill_tokens"] = 16
    rows[2]["cached_tokens"] = 113
    rows[2]["last_cache_execution"]["cached_tokens"] = 113
    rows[2]["last_cache_execution"]["attempted_cached_tokens"] = 113
    rows[2]["last_cache_execution"]["uncached_prompt_tokens"] = 15
    rows[2]["last_cache_execution"]["prefill_tokens"] = 15

    failures = _validate_rows("store", rows)

    assert any(
        "not a longest continuous shared-prefix result" in item for item in failures
    )
    assert rows[2]["cache_contract_ok"] is False


def test_probe_linkage_accepts_same_source_config_and_new_engine():
    assert (
        validate_probe_linkage(
            _probe_metadata(),
            _store_summary(),
        )
        == []
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", "localhost"),
        ("port", 8002),
        ("command", "python -m vmlx_engine.server --port 8002"),
    ],
)
def test_probe_linkage_requires_stable_listener_endpoint_and_command(field, value):
    probe = _probe_metadata()
    probe["identity"]["observed_engine"][field] = value

    failures = validate_probe_linkage(probe, _store_summary())

    assert any(
        f"observed listener {field} does not match store" in failure
        for failure in failures
    )


@pytest.mark.parametrize(
    ("probe_changes", "store_changes", "expected_failure"),
    [
        (
            {"engine": "engine-before-restart"},
            {},
            "observed listener identity must differ",
        ),
        ({"source": "other-source"}, {}, "declared source identity does not match"),
        ({"config": "other-config"}, {}, "declared config identity does not match"),
        ({"nonce": "other-nonce"}, {}, "nonce does not match"),
        ({"model": "other/model"}, {}, "model does not match"),
        ({}, {"cache_contract_ok": False}, "store summary cache contract did not pass"),
        ({}, {"gate_ok": False}, "store summary full gate did not pass"),
        (
            {},
            {"store_durability": {"ok": False}},
            "store durability barrier did not pass",
        ),
    ],
)
def test_probe_linkage_rejects_identity_or_store_contract_mismatch(
    probe_changes, store_changes, expected_failure
):
    probe = _probe_metadata()
    store = _store_summary()
    for key, value in probe_changes.items():
        if key == "engine":
            probe["identity"]["observed_engine"] = _observed_engine(value)
        elif key == "source":
            probe["identity"]["declared_source"] = value
            probe["identity"]["observed_source"] = _observed_source(value)
        elif key == "config":
            probe["identity"]["declared_config"] = value
        else:
            probe[key] = value
    store.update(store_changes)

    failures = validate_probe_linkage(probe, store)

    assert any(expected_failure in failure for failure in failures)


def test_probe_linkage_rejects_different_common_prefix_contract():
    probe = _probe_metadata()
    probe["prompt_contract"] = _prompt_contract("different-prefix", 320)

    failures = validate_probe_linkage(probe, _store_summary())

    assert any("prompt contract does not match" in failure for failure in failures)


def test_probe_main_rejects_same_engine_before_sending_requests(monkeypatch, tmp_path):
    store_path = tmp_path / "store-summary.json"
    store_path.write_text(json.dumps(_store_summary()))

    def unexpected_post(*_args, **_kwargs):
        raise AssertionError("probe request must not run with invalid linkage")

    monkeypatch.setattr(gate, "_post_sse", unexpected_post)
    monkeypatch.setattr(gate, "_json_get", lambda _url, _timeout: _health())
    monkeypatch.setattr(
        gate,
        "_fetch_tokenizer_lcp_contract",
        lambda **_kwargs: (TOKEN_CONTRACT, []),
    )
    monkeypatch.setattr(
        gate,
        "_observe_local_listener_identity",
        lambda _base_url: _observed_engine("engine-before-restart"),
    )
    monkeypatch.setattr(
        gate,
        "_observe_source_checkout",
        lambda: _observed_source(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cache_hierarchy_live_gate.py",
            "--model",
            MODEL,
            "--nonce",
            NONCE,
            "--source-identity",
            SOURCE,
            "--config-identity",
            CONFIG,
            "--store-summary",
            str(store_path),
            "--artifact-dir",
            str(tmp_path / "probe"),
            "--phase",
            "probe",
        ],
    )

    exit_code = gate.main()
    summary = json.loads((tmp_path / "probe" / "summary.json").read_text())

    assert exit_code == 1
    assert summary["requests"] == []
    assert summary["probe_linkage_ok"] is False
    assert any(
        "observed listener identity must differ" in failure
        for failure in summary["probe_linkage_failures"]
    )


def test_observed_listener_identity_uses_unique_lsof_pid_and_ps_provenance(
    monkeypatch,
):
    def fake_run_text(command):
        if command[0] == "/usr/sbin/lsof":
            if "-iTCP:8001" in command:
                return "p4321"
            if "cwd" in command:
                return f"p4321\nfcwd\nn{GIT_ROOT}"
            raise AssertionError(command)
        if command[-1] == "lstart=":
            return "Fri Jul 24 10:00:00 2026"
        if command[-1] == "command=":
            return f"{TEST_PYTHON} -m vmlx_engine.cli serve test/model --port 8001"
        raise AssertionError(command)

    monkeypatch.setattr(gate, "_run_text", fake_run_text)

    identity = _observe_local_listener_identity("http://127.0.0.1:8001")

    assert identity["method"] == "macos-lsof-ps"
    assert identity["pid"] == 4321
    assert identity["started_at"] == "Fri Jul 24 10:00:00 2026"
    assert identity["command"].endswith("--port 8001")
    assert identity["cwd"] == str(GIT_ROOT)
    assert identity["python_executable"] == TEST_PYTHON
    assert identity["launch_shape"] == "python-module-vmlx-engine-cli"
    assert len(identity["fingerprint_sha256"]) == 64


def test_observed_listener_identity_rejects_ambiguous_listener_pids(monkeypatch):
    monkeypatch.setattr(gate, "_run_text", lambda _command: "p4321\np9876")

    with pytest.raises(RuntimeError, match="expected one LISTEN PID"):
        _observe_local_listener_identity("http://localhost:8001")


def test_health_runtime_provenance_binds_listener_pid_and_source_hashes():
    assert (
        _validate_health_runtime_provenance(
            _health(),
            _observed_engine("engine"),
            _observed_source(),
        )
        == []
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pid", 9876),
        ("server_module_sha256", "not-current-source"),
        ("python_source_tree_sha256", "other-checkout"),
        ("python_source_read_error_count", 1),
    ],
)
def test_health_runtime_provenance_rejects_listener_or_source_mismatch(
    field,
    value,
):
    health = _health()
    health["runtime_provenance"][field] = value

    failures = _validate_health_runtime_provenance(
        health,
        _observed_engine("engine"),
        _observed_source(),
    )

    assert any(field in failure for failure in failures)


def test_artifact_dir_rejects_worktree_paths_and_symlinks_into_worktree(tmp_path):
    git_root = Path(gate.__file__).resolve().parents[2]

    with pytest.raises(RuntimeError, match="outside every Git"):
        _external_artifact_dir(git_root / "private-live-evidence")

    link = tmp_path / "evidence-link"
    link.symlink_to(git_root / "tests", target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside every Git"):
        _external_artifact_dir(link / "raw-sse")

    assert (
        _external_artifact_dir(tmp_path / "external-evidence")
        == (tmp_path / "external-evidence").resolve()
    )


def test_artifact_dir_rejects_sibling_git_repository(tmp_path):
    sibling_repo = tmp_path / "sibling-public-repo"
    subprocess.run(
        ["/usr/bin/git", "init", "-q", str(sibling_repo)],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match="outside every Git"):
        _external_artifact_dir(sibling_repo / "private-live-evidence")

    assert not (sibling_repo / "private-live-evidence").exists()


def test_main_rejects_worktree_artifact_dir_before_observing_runtime(monkeypatch):
    git_root = Path(gate.__file__).resolve().parents[2]
    monkeypatch.setattr(
        gate,
        "_observe_local_listener_identity",
        lambda _base_url: pytest.fail("runtime provenance must not be observed"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_cache_hierarchy_live_gate.py",
            "--model",
            MODEL,
            "--nonce",
            NONCE,
            "--source-identity",
            SOURCE,
            "--config-identity",
            CONFIG,
            "--artifact-dir",
            str(git_root / "tests"),
            "--phase",
            "store",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        gate.main()

    assert exc_info.value.code == 2


def _disk_health(
    *,
    writes: int,
    blocks: int,
    request_id: str | None = "resp-cold_a",
    expected: int = 1,
    queued: int | None = None,
    completed: int | None = None,
    retained: int | None = None,
    failed: int = 0,
    dropped: int = 0,
    post_eviction_complete: bool = True,
    queue_depth: int = 0,
    inflight: int = 0,
    completion_generation: int = 1,
    evictions: int = 0,
    strict_fences: bool = True,
    reconciliation_generation: int = 1,
    accounting_generation: int = 1,
    managed_bytes: int = 900,
    managed_max_bytes: int = 1000,
) -> dict:
    fences = []
    if request_id is not None:
        fences.append(
            {
                "fence_id": "block-write-1",
                "request_id": request_id,
                "expected": expected,
                "queued": expected if queued is None else queued,
                "completed": expected if completed is None else completed,
                "failed": failed,
                "dropped": dropped,
                "retained": expected if retained is None else retained,
                "sealed": True,
                "seal_enqueued": True,
                "seal_failed": False,
                "post_eviction_complete": post_eviction_complete,
                "completion_generation": completion_generation,
                "global_reconciliation_generation": reconciliation_generation,
                "global_accounting_generation": accounting_generation,
                "global_bytes_after": managed_bytes,
                "global_max_size_bytes": managed_max_bytes,
            }
        )
    return {
        "scheduler": {"num_waiting": 0, "num_running": 0},
        "cache": {
            "scheduler_cache": {
                "strict_block_disk_write_fence": strict_fences,
            },
            "block_disk_cache": {
                "disk_writes": writes,
                "disk_evictions": evictions,
                "blocks_on_disk": blocks,
                "global_budget": {
                    "accounted": True,
                    "compliant": (
                        managed_max_bytes <= 0 or managed_bytes <= managed_max_bytes
                    ),
                    "bytes_after": managed_bytes,
                    "max_size_bytes": managed_max_bytes,
                    "accounting_generation": accounting_generation,
                    "reconciliation_generation": reconciliation_generation,
                },
                "write_pipeline": {
                    "queue_depth": queue_depth,
                    "inflight": inflight,
                    "writer_alive": True,
                    "completion_generation": completion_generation,
                    "recent_fences": fences,
                },
            }
        },
    }


def test_store_durability_barrier_accepts_full_capacity_replacement(
    monkeypatch,
):
    baseline = _health_cache_counters(
        _disk_health(writes=10, blocks=5, reconciliation_generation=0)
    )
    health_responses = iter(
        [
            _disk_health(writes=10, blocks=5, request_id=None),
            _disk_health(writes=11, blocks=5, evictions=1),
            _disk_health(writes=11, blocks=5, evictions=1),
        ]
    )
    monkeypatch.setattr(
        gate,
        "_json_get",
        lambda _url, _timeout: next(health_responses),
    )

    durability, final_health = _wait_for_store_durability(
        base_url="http://127.0.0.1:8001",
        request_timeout=1,
        request_id="resp-cold_a",
        baseline_counters=baseline,
        timeout_s=1.0,
        poll_interval_s=0.001,
    )

    assert durability["ok"] is True
    assert durability["polls"] == 3
    assert durability["stable_observations"] == 2
    assert durability["matching_fence"]["retained"] == 1
    assert durability["counter_deltas"]["block_disk_cache.disk_writes"] == 1
    assert durability["counter_deltas"]["block_disk_cache.blocks_on_disk"] == 0
    assert final_health == _disk_health(writes=11, blocks=5, evictions=1)


def test_store_durability_barrier_is_request_correlated_not_scheduler_global(
    monkeypatch,
):
    baseline = _health_cache_counters(
        _disk_health(writes=10, blocks=5, reconciliation_generation=0)
    )
    first_health = _disk_health(writes=11, blocks=6)
    first_health["scheduler"] = {"num_waiting": 1, "num_running": 1}
    final_health = _disk_health(writes=11, blocks=6)
    final_health["scheduler"] = {"num_waiting": 2, "num_running": 3}
    health_responses = iter([first_health, final_health])
    monkeypatch.setattr(
        gate,
        "_json_get",
        lambda _url, _timeout: next(health_responses),
    )

    durability, final_health = _wait_for_store_durability(
        base_url="http://127.0.0.1:8001",
        request_timeout=1,
        request_id="resp-cold_a",
        baseline_counters=baseline,
        timeout_s=1.0,
        poll_interval_s=0.001,
    )

    assert durability["ok"] is True
    assert durability["stable_observations"] == 2
    assert durability["matching_fence"]["request_id"] == "resp-cold_a"
    assert final_health["scheduler"] == {"num_waiting": 2, "num_running": 3}


def test_store_durability_barrier_rejects_aggregate_write_without_request_fence(
    monkeypatch,
):
    baseline = _health_cache_counters(
        _disk_health(writes=10, blocks=5, reconciliation_generation=0)
    )
    monkeypatch.setattr(
        gate,
        "_json_get",
        lambda _url, _timeout: _disk_health(
            writes=11,
            blocks=5,
            request_id="resp-other",
        ),
    )

    durability, _final_health = _wait_for_store_durability(
        base_url="http://127.0.0.1:8001",
        request_timeout=1,
        request_id="resp-cold_a",
        baseline_counters=baseline,
        timeout_s=0,
        poll_interval_s=0.001,
    )

    assert durability["ok"] is False
    assert durability["polls"] == 1
    assert durability["counter_deltas"]["block_disk_cache.disk_writes"] == 1
    assert durability["counter_deltas"]["block_disk_cache.blocks_on_disk"] == 0
    assert any(
        "no block-disk write fence matches" in item
        for item in durability["contract_failures"]
    )


def test_store_durability_barrier_rejects_post_eviction_loss(monkeypatch):
    baseline = _health_cache_counters(
        _disk_health(writes=10, blocks=5, reconciliation_generation=0)
    )
    monkeypatch.setattr(
        gate,
        "_json_get",
        lambda _url, _timeout: _disk_health(
            writes=11,
            blocks=5,
            expected=2,
            retained=1,
            evictions=2,
        ),
    )

    durability, _final_health = _wait_for_store_durability(
        base_url="http://127.0.0.1:8001",
        request_timeout=1,
        request_id="resp-cold_a",
        baseline_counters=baseline,
        timeout_s=0,
        poll_interval_s=0.001,
    )

    assert durability["ok"] is False
    assert any(
        "retained=1 != expected=2" in item for item in durability["contract_failures"]
    )


@pytest.mark.parametrize(
    ("health_kwargs", "expected_failure"),
    [
        (
            {"strict_fences": False},
            "not launched with strict physical block-disk fences",
        ),
        (
            {"reconciliation_generation": 0},
            "did not advance physical reconciliation",
        ),
        (
            {"managed_bytes": 1001, "managed_max_bytes": 1000},
            "managed-root physical bytes are over limit",
        ),
    ],
)
def test_store_durability_barrier_requires_strict_physical_attestation(
    monkeypatch,
    health_kwargs,
    expected_failure,
):
    baseline = _health_cache_counters(
        _disk_health(writes=10, blocks=5, reconciliation_generation=0)
    )
    monkeypatch.setattr(
        gate,
        "_json_get",
        lambda _url, _timeout: _disk_health(
            writes=11,
            blocks=6,
            **health_kwargs,
        ),
    )

    durability, _final_health = _wait_for_store_durability(
        base_url="http://127.0.0.1:8001",
        request_timeout=1,
        request_id="resp-cold_a",
        baseline_counters=baseline,
        timeout_s=0,
        poll_interval_s=0.001,
    )

    assert durability["ok"] is False
    assert any(
        expected_failure in failure
        for failure in durability["contract_failures"]
    )


def test_health_cache_counters_retain_request_local_disk_hit_delta():
    before = {
        "scheduler": {"cache_hit_requests": 3, "cache_hit_tokens": 256},
        "cache": {
            "scheduler_cache": {
                "cache_hits": 4,
                "disk_hits": 7,
                "tokens_saved": 256,
            },
            "block_disk_cache": {
                "disk_hits": 7,
                "disk_writes": 12,
                "blocks_on_disk": 12,
                "total_tokens_on_disk": 768,
            },
        },
    }
    after = deepcopy(before)
    after["scheduler"]["cache_hit_requests"] = 4
    after["scheduler"]["cache_hit_tokens"] = 368
    after["cache"]["scheduler_cache"]["cache_hits"] = 6
    after["cache"]["scheduler_cache"]["disk_hits"] = 9
    after["cache"]["scheduler_cache"]["tokens_saved"] = 368
    after["cache"]["block_disk_cache"]["disk_hits"] = 9

    counters_before = _health_cache_counters(before)
    counters_after = _health_cache_counters(after)
    deltas = _counter_deltas(counters_before, counters_after)

    assert counters_before["block_disk_cache.blocks_on_disk"] == 12
    assert deltas["scheduler.cache_hit_requests"] == 1
    assert deltas["scheduler.cache_hit_tokens"] == 112
    assert deltas["scheduler_cache.disk_hits"] == 2
    assert deltas["block_disk_cache.disk_hits"] == 2
