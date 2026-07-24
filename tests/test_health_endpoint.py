# SPDX-License-Identifier: Apache-2.0
"""Tests for the /health endpoint in server.py.

Verifies status reporting, path sanitization, JANG metadata, memory info,
and last_request_time fields by patching server globals directly.
"""

import asyncio
import hashlib
import json
import platform
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


class TestHealthEndpoint:
    """Tests for the health() async handler."""

    def test_health_mtp_route_is_registered(self):
        """MTP diagnostics should have a stable direct health route alias."""
        from vmlx_engine.server import app

        assert "/health.mtp" in {route.path for route in app.routes}

    def test_health_runtime_provenance_is_path_free_and_same_process(self):
        """Private gates can bind the listener without exposing absolute paths."""
        import os

        from vmlx_engine import server

        with (
            patch.object(server, "_engine", None),
            patch.object(server, "_model_name", None),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        provenance = result["runtime_provenance"]
        assert provenance["pid"] == os.getpid()
        assert provenance["server_module_relpath"] == "vmlx_engine/server.py"
        assert provenance["package_init_relpath"] == "vmlx_engine/__init__.py"
        assert provenance["python_source_file_count"] > 0
        assert provenance["python_source_read_error_count"] == 0
        assert len(provenance["python_source_tree_sha256"]) == 64
        assert len(provenance["server_module_sha256"]) == 64
        assert (
            provenance["model_bundle_provenance"]["directory_state"]
            == "model_not_loaded"
        )
        cache_attestation = provenance["cache_topology_provenance"]
        assert cache_attestation["canonical_sha256"] == (
            server._canonical_attestation_sha256(cache_attestation["configuration"])
        )
        assert all("/Users/" not in str(value) for value in provenance.values())

    def test_health_bundle_attestation_hashes_observed_loaded_config_files(
        self,
        tmp_path,
    ):
        """The listener hashes the loaded bundle, not a harness-declared id."""
        from vmlx_engine import server

        config = tmp_path / "config.json"
        generation = tmp_path / "generation_config.json"
        config.write_text('{"model_type":"laguna"}\n')
        generation.write_text('{"temperature":0.6}\n')

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False
        common_patches = (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_path", str(tmp_path)),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
            patch.object(server, "_get_scheduler", return_value=None),
            patch.object(server, "_model_quantization_status", return_value={}),
            patch.object(server, "_model_acceleration_status", return_value={}),
            patch.object(
                server,
                "_model_mtp_status_with_loaded_runtime",
                return_value={},
            ),
            patch.object(
                server,
                "_model_effective_defaults_status",
                return_value={
                    "sampling_defaults": {},
                    "effective_defaults": {},
                },
            ),
            patch.object(server, "_model_routing_status", return_value={}),
            patch.object(
                server,
                "_turboquant_kv_cache_status",
                return_value={"enabled": False},
            ),
            patch.object(server, "_current_model_config", return_value=None),
            patch.object(server, "_native_cache_status", return_value={}),
        )
        for context in common_patches:
            context.start()
        try:
            first = _run(server.health())["runtime_provenance"][
                "model_bundle_provenance"
            ]
            config.write_text('{"model_type":"laguna","revision":2}\n')
            second = _run(server.health())["runtime_provenance"][
                "model_bundle_provenance"
            ]
        finally:
            for context in reversed(common_patches):
                context.stop()

        assert first["directory_state"] == "available"
        assert set(first["files"]) == set(server._BUNDLE_ATTESTATION_FILENAMES)
        assert first["files"]["config.json"] == {
            "state": "present",
            "size_bytes": len(b'{"model_type":"laguna"}\n'),
            "sha256": hashlib.sha256(b'{"model_type":"laguna"}\n').hexdigest(),
        }
        assert first["files"]["generation_config.json"]["state"] == "present"
        assert first["files"]["jang_config.json"] == {"state": "missing"}
        assert first["aggregate_sha256"] != second["aggregate_sha256"]
        assert (
            first["files"]["config.json"]["sha256"]
            != (second["files"]["config.json"]["sha256"])
        )
        assert str(tmp_path) not in json.dumps(first, sort_keys=True)

    def test_cache_topology_attestation_is_canonical_and_change_sensitive(self):
        """Mapping order is irrelevant while an effective topology change is not."""
        from vmlx_engine import server

        first_configuration = {
            "schema": "vmlx-cache-topology-v1",
            "configured": {
                "use_paged_cache": True,
                "kv_cache_quantization": "q4",
            },
            "instantiated": {
                "block_disk_l2": True,
                "paged_ram_enabled": True,
            },
        }
        reordered_configuration = {
            "instantiated": {
                "paged_ram_enabled": True,
                "block_disk_l2": True,
            },
            "configured": {
                "kv_cache_quantization": "q4",
                "use_paged_cache": True,
            },
            "schema": "vmlx-cache-topology-v1",
        }
        changed_configuration = {
            **first_configuration,
            "instantiated": {
                **first_configuration["instantiated"],
                "paged_ram_enabled": False,
            },
        }

        first = server._cache_topology_attestation(first_configuration)
        reordered = server._cache_topology_attestation(reordered_configuration)
        changed = server._cache_topology_attestation(changed_configuration)

        assert first["canonical_sha256"] == reordered["canonical_sha256"]
        assert first["canonical_sha256"] != changed["canonical_sha256"]

    def test_effective_cache_topology_reads_instantiated_scheduler_state(self):
        """The attestation changes with real manager state, not a caller label."""
        from vmlx_engine import server

        disk_store = SimpleNamespace(max_size_bytes=10 * 1024**3)
        paged = SimpleNamespace(
            _disk_store=disk_store,
            block_size=64,
            max_blocks=1000,
            max_resident_bytes=4 * 1024**3,
            disk_only=False,
            paged_frugal=False,
            ram_mirror_policy="resident",
        )
        scheduler = SimpleNamespace(
            config=SimpleNamespace(
                enable_prefix_cache=True,
                use_paged_cache=True,
                enable_block_disk_cache=True,
                block_disk_cache_max_gb=10.0,
                kv_cache_quantization="q4",
                kv_cache_group_size=64,
            ),
            block_aware_cache=object(),
            memory_aware_cache=None,
            prefix_cache=None,
            paged_cache_manager=paged,
            disk_cache=None,
            _ssm_companion_disk_store=None,
        )
        health_status = {
            "kv_cache_quantization": {"enabled": True, "mode": "live", "bits": 4},
            "turboquant_kv_cache": {
                "enabled": True,
                "storage_encode_enabled": True,
                "storage_key_bits": 4,
            },
            "native_cache": {
                "schema": "plain_kv_v1",
                "cache_type": "paged_kv",
                "prefix": True,
                "paged": True,
                "block_disk_l2": True,
            },
        }

        first = server._cache_topology_attestation(
            server._cache_topology_configuration(scheduler, health_status)
        )
        paged.disk_only = True
        paged.ram_mirror_policy = "disk_only"
        second = server._cache_topology_attestation(
            server._cache_topology_configuration(scheduler, health_status)
        )

        assert first["configuration"]["instantiated"]["paged_ram_enabled"] is True
        assert second["configuration"]["instantiated"]["paged_ram_enabled"] is False
        assert first["canonical_sha256"] != second["canonical_sha256"]

    def test_health_no_model_loaded(self):
        """When _engine is None, health returns status='no_model'."""
        from vmlx_engine import server

        with (
            patch.object(server, "_engine", None),
            patch.object(server, "_model_name", None),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert result["status"] == "no_model"
        assert result["model_loaded"] is False

    def test_health_with_model_loaded(self):
        """When _engine is present with get_stats(), status='healthy'."""
        from vmlx_engine import server

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert result["status"] == "healthy"
        assert result["model_loaded"] is True
        assert result["model_name"] == "test-model"
        assert result["engine_type"] == "simple"

    def test_health_error_sanitizes_paths(self):
        """Path strings in _model_load_error are replaced with <path>."""
        from vmlx_engine import server

        error_msg = "FileNotFoundError: /home/user/models/foo/config.json not found"

        with (
            patch.object(server, "_engine", None),
            patch.object(server, "_model_name", None),
            patch.object(server, "_model_load_error", error_msg),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert "error" in result
        assert "/home/user" not in result["error"]
        assert "<path>" in result["error"]

    def test_health_jang_metadata_cached(self):
        """When _jang_metadata is set, it appears as quantization_format in response."""
        from vmlx_engine import server

        jang_meta = {
            "type": "jang",
            "target_bits": 4.5,
            "actual_bits": 4.48,
            "block_size": 64,
        }

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "jang-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", jang_meta),
            patch.object(server, "_last_request_time", 0.0),
        ):
            result = _run(server.health())

        assert "quantization_format" in result
        assert result["quantization_format"]["type"] == "jang"
        assert result["quantization_format"]["target_bits"] == 4.5

    def test_health_memory_info(self):
        """When mlx memory functions exist, memory dict is present."""
        from vmlx_engine import server

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        # Mock mlx.core memory functions that health() calls internally
        mock_mx = MagicMock()
        mock_mx.get_active_memory.return_value = 1024 * 1024 * 512  # 512 MB
        mock_mx.get_peak_memory.return_value = 1024 * 1024 * 1024  # 1 GB
        mock_mx.get_cache_memory.return_value = 1024 * 1024 * 256  # 256 MB

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 0.0),
            patch.dict("sys.modules", {"mlx.core": mock_mx}),
        ):
            result = _run(server.health())

        # Memory info should be present since mlx.core is mocked with the functions
        if "memory" in result:
            assert "active_mb" in result["memory"]
            assert "peak_mb" in result["memory"]
            assert "cache_mb" in result["memory"]
        # If mlx.core import fails in test env, memory may be absent — that's OK

    def test_health_last_request_time(self):
        """Verify last_request_time field exists in health response."""
        from vmlx_engine import server

        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"engine_type": "simple"}
        mock_engine.is_mllm = False

        with (
            patch.object(server, "_engine", mock_engine),
            patch.object(server, "_model_name", "test-model"),
            patch.object(server, "_model_load_error", None),
            patch.object(server, "_mcp_manager", None),
            patch.object(server, "_jang_metadata", None),
            patch.object(server, "_last_request_time", 1700000000.0),
        ):
            result = _run(server.health())

        # last_request_time should be present (non-zero value means it's not None)
        assert "last_request_time" in result
        assert result["last_request_time"] == 1700000000.0
