# SPDX-License-Identifier: Apache-2.0
"""Contracts for the DSV4 Responses restart/L2 gate preflight."""

import sys
from pathlib import Path
from types import SimpleNamespace

from tests.cross_matrix import run_dsv4_responses_restart_l2_gate as gate


def test_dsv4_responses_restart_l2_resource_snapshot_labels_binary_units(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(
                total=128 * 1024**3,
                available=74 * 1024**3,
                percent=42.0,
            )
        ),
    )

    snapshot = gate.resource_snapshot("preflight")

    assert snapshot["system_memory"]["unit"] == "GiB"
    assert snapshot["system_memory"]["total_gib"] == 128.0
    assert snapshot["system_memory"]["available_gib"] == 74.0
    assert snapshot["system_memory"]["total_gb"] == 128.0
    assert snapshot["system_memory"]["available_gb"] == 74.0


def test_dsv4_responses_restart_l2_memory_preflight_labels_binary_units(monkeypatch):
    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name: {
            "name": name,
            "system_memory": {
                "unit": "GiB",
                "available_gib": 74.0,
                "available_gb": 74.0,
                "total_gib": 128.0,
                "total_gb": 128.0,
            },
        },
    )
    args = SimpleNamespace(min_free_gb=80.0)

    artifact = gate.blocked_by_memory_preflight(args)

    assert artifact is not None
    assert artifact["status"] == "skipped"
    assert artifact["reason"] == "insufficient_free_memory"
    assert artifact["unit"] == "GiB"
    assert artifact["required_available_gib"] == 80.0
    assert artifact["required_available_gb"] == 80.0
    assert artifact["available_gib"] == 74.0
    assert artifact["available_gb"] == 74.0
    assert artifact["memory_gap_gib"] == 6.0
    assert artifact["memory_gap_gb"] == 6.0
    assert artifact["telemetry"][0]["system_memory"]["available_gib"] == 74.0


def test_restart_gate_launches_ssd_only_with_percent_budget_and_private_contract():
    args = SimpleNamespace(
        python=Path("/tmp/python3"),
        model="/tmp/dsv4",
        port=8843,
        max_output_tokens=192,
        block_disk_cache_percent=3.5,
    )

    command = gate.build_command(
        args,
        Path("/tmp/block-cache"),
        "dsv4-restart-l2",
        Path("/tmp/private.token"),
    )

    assert "--no-paged-cache" in command
    assert "--use-paged-cache" not in command
    assert "--block-disk-cache-max-gb" not in command
    assert command[command.index("--block-disk-cache-max-percent") + 1] == "3.5"
    assert "--enable-private-cache-attestation" in command
    assert (
        command[command.index("--private-cache-attestation-token-file") + 1]
        == "/tmp/private.token"
    )
    assert Path(
        command[command.index("--private-cache-attestation-token-file") + 1]
    ).is_absolute()


def test_restart_gate_selects_production_off_boundary_prompt(monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout=600, *, headers=None):
        captured.update(
            url=url,
            payload=payload,
            timeout=timeout,
            headers=headers,
        )
        label = next(iter(payload["inputs"]))
        index = int(label.rsplit("_", 1)[1])
        rows = {
            label: {
                "cache_prompt_token_count": 512 if index == 0 else 513,
                "cache_prompt_token_ids_sha256": "a" * 64,
            }
        }
        return {
            "method": "final-render-tokenize-no-cache",
            "surface": "responses",
            "cache_lookup_bypassed": True,
            "prompts": rows,
        }

    monkeypatch.setattr(gate, "post_json", fake_post)

    prompt, row, contract = gate.select_off_boundary_prompt(
        base_url="http://127.0.0.1:8843",
        model_name="dsv4-restart-l2",
        long_context="context",
        proof_token="x" * 48,
        timeout=9,
    )

    assert row["cache_prompt_token_count"] == 513
    assert "BOUNDARY PAD 1" in prompt
    assert contract["cache_lookup_bypassed"] is True
    assert captured["url"].endswith("/v1/cache/token-contract")
    assert captured["payload"]["request_controls"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert captured["headers"]["Authorization"] == "Bearer " + ("x" * 48)
    assert (
        captured["headers"]["X-vMLX-Private-Proof"]
        == gate.PRIVATE_ATTESTATION_PROOF_HEADER
    )


def test_restart_gate_waits_for_its_async_disk_write_fence(monkeypatch):
    snapshots = iter(
        [
            {
                "block_disk_cache": {
                    "disk_writes": 0,
                    "blocks_on_disk": 0,
                    "write_pipeline": {
                        "active_producers": 1,
                        "inflight": 1,
                        "pending_items": 1,
                        "pending_bytes": 1024,
                    },
                }
            },
            {
                "block_disk_cache": {
                    "disk_writes": 21,
                    "blocks_on_disk": 21,
                    "write_pipeline": {
                        "active_producers": 0,
                        "inflight": 0,
                        "pending_items": 0,
                        "pending_bytes": 0,
                    },
                }
            },
        ]
    )
    monkeypatch.setattr(gate, "get_json", lambda *_args, **_kwargs: next(snapshots))
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)

    durable = gate.wait_block_disk_durable("http://127.0.0.1:8843", timeout_s=1)

    block = durable["block_disk_cache"]
    assert block["disk_writes"] == 21
    assert block["blocks_on_disk"] == 21
    assert block["write_pipeline"]["inflight"] == 0
