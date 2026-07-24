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
    _compare_attestation_snapshots,
    _compare_source_checkout_observations,
    _counter_deltas,
    _external_artifact_dir,
    _health_attestation_snapshot,
    _health_cache_counters,
    _observe_local_listener_identity,
    _prompt_contract,
    _summarize,
    _token_contract_request,
    _validate_health_runtime_provenance,
    _validate_tokenizer_lcp_contract,
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
        }
        for label in ("A", "B", "C")
    },
    "longest_common_prefix_tokens": {
        "A:A": 128,
        "A:B": 127,
        "A:C": 127,
    },
}


def _validate_rows(
    phase: str,
    rows: list[dict],
    *,
    store_summary: dict | None = None,
    token_contract: dict | None = None,
) -> list[str]:
    return validate_cache_rows(
        phase,
        rows,
        store_summary=store_summary,
        token_contract=TOKEN_CONTRACT if token_contract is None else token_contract,
    )


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


def test_store_contract_accepts_cold_warm_and_longest_partial_prefix():
    rows = _valid_store_rows()

    failures = _validate_rows("store", rows)

    assert failures == []
    assert all(row["cache_contract_ok"] is True for row in rows)
    assert rows[2]["expected_shared_prefix_floor_tokens"] == 112
    assert rows[2]["last_cache_execution"]["prefill_tokens"] == 16


def test_standard_scheduler_prefill_does_not_require_generation_suffix_field():
    rows = _valid_store_rows()

    assert "generation_prompt_suffix_tokens" not in rows[2]["last_cache_execution"]
    assert _validate_rows("store", rows) == []


def test_mllm_prefill_includes_generation_prompt_suffix_tokens():
    rows = _valid_store_rows()
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
    assert _validate_rows("store", rows) == []


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
    monkeypatch.setattr(
        gate,
        "_summarize",
        lambda _raw, _elapsed, _status: {
            "status_code": 200,
            "elapsed_s": 0.01,
            "event_counts": {"response.completed": 1},
            "terminal_events": ["response.completed"],
            "output_text": ("CACHE-HIERARCHY-http-only-A CACHE-HIERARCHY-http-only-B"),
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
            }
        )
    return {
        "scheduler": {"num_waiting": 0, "num_running": 0},
        "cache": {
            "block_disk_cache": {
                "disk_writes": writes,
                "disk_evictions": evictions,
                "blocks_on_disk": blocks,
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
    baseline = _health_cache_counters(_disk_health(writes=10, blocks=5))
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


def test_store_durability_barrier_rejects_aggregate_write_without_request_fence(
    monkeypatch,
):
    baseline = _health_cache_counters(_disk_health(writes=10, blocks=5))
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
    baseline = _health_cache_counters(_disk_health(writes=10, blocks=5))
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
