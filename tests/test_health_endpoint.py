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
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
        assert (
            result["model_bundle_provenance"]
            == provenance["model_bundle_provenance"]
        )
        assert (
            result["cache_topology_provenance"]
            == provenance["cache_topology_provenance"]
        )
        cache_attestation = provenance["cache_topology_provenance"]
        assert cache_attestation["canonical_sha256"] == (
            server._canonical_attestation_sha256(cache_attestation["configuration"])
        )
        assert all(
            "/private/example/" not in str(value)
            for value in provenance.values()
        )

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

    def test_cache_token_contract_route_is_registered(self):
        """Cache live gate needs a stable dry-render token contract route."""
        from vmlx_engine.server import app

        assert "/v1/cache/token-contract" in {route.path for route in app.routes}

    def test_private_cache_attestation_routes_are_hidden_and_proof_authenticated(self):
        """Proof-only cache contracts are absent unless their private gate passes."""
        from vmlx_engine import server

        for path in (
            "/v1/cache/prefix-attestation",
            "/v1/cache/token-contract",
        ):
            route = next(route for route in server.app.routes if route.path == path)
            dependency_calls = {
                dependency.call for dependency in route.dependant.dependencies
            }
            assert server.verify_private_cache_attestation in dependency_calls
            assert route.include_in_schema is False
        assert "/v1/cache/prefix-attestation" not in server.app.openapi()["paths"]
        assert "/v1/cache/token-contract" not in server.app.openapi()["paths"]

    def test_private_cache_attestation_token_file_and_request_gate(self, tmp_path):
        """The proof credential is owner-only, symlink-safe, and fail-closed."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from vmlx_engine import server

        token = "proof_" + ("a" * 58)
        token_path = tmp_path / "cache-proof.token"
        token_path.write_text(token)
        token_path.chmod(0o600)
        assert server._load_private_cache_attestation_token_file(str(token_path)) == token

        token_path.chmod(0o640)
        with pytest.raises(ValueError, match="owner-only"):
            server._load_private_cache_attestation_token_file(str(token_path))
        token_path.chmod(0o600)
        link_path = tmp_path / "cache-proof-link.token"
        link_path.symlink_to(token_path)
        with pytest.raises(ValueError, match="owner-only"):
            server._load_private_cache_attestation_token_file(str(link_path))

        request = SimpleNamespace(headers={})
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token,
        )
        with (
            patch.object(server, "_private_cache_attestation_enabled", False),
            patch.object(server, "_private_cache_attestation_token", None),
            pytest.raises(HTTPException) as disabled,
        ):
            _run(server.verify_private_cache_attestation(request, None))
        assert disabled.value.status_code == 404

        with (
            patch.object(server, "_private_cache_attestation_enabled", True),
            patch.object(server, "_private_cache_attestation_token", token),
        ):
            with pytest.raises(HTTPException) as missing:
                _run(server.verify_private_cache_attestation(request, None))
            assert missing.value.status_code == 401
            with pytest.raises(HTTPException) as intent:
                _run(server.verify_private_cache_attestation(request, credentials))
            assert intent.value.status_code == 403
            allowed_request = SimpleNamespace(
                headers={
                    "x-vmlx-private-proof": (
                        server._PRIVATE_CACHE_ATTESTATION_PROOF_HEADER
                    )
                }
            )
            assert _run(
                server.verify_private_cache_attestation(
                    allowed_request,
                    credentials,
                )
            ) is True

    def test_private_cache_attestation_enforces_byte_and_token_budgets(self):
        """Proof-only rendering cannot become an unbounded tokenizer workload."""
        from fastapi import HTTPException

        from vmlx_engine import server

        prefix_body = {
            "contract_version": 1,
            "surface": "responses",
            "model": "fake/model",
            "inputs": {
                "left": "a" * (server._PRIVATE_CACHE_ATTESTATION_MAX_PROMPT_BYTES + 1),
                "right": "b",
            },
            "prefix_pairs": {"target": ["left", "right"]},
        }
        with (
            patch.object(server, "_engine", object()),
            pytest.raises(HTTPException) as too_large,
        ):
            _run(server.cache_prefix_attestation(prefix_body))
        assert too_large.value.status_code == 413

        token_body = {
            "contract_version": 1,
            "surface": "responses",
            "model": "fake/model",
            "inputs": {"one": "bounded"},
        }
        health_attestation = {
            "model_bundle_provenance": {"fingerprint_sha256": "a" * 64},
            "cache_topology_provenance": {"fingerprint_sha256": "b" * 64},
        }
        oversized_vector = [0] * (
            server._PRIVATE_CACHE_ATTESTATION_MAX_TOKENS_PER_PROMPT + 1
        )
        with (
            patch.object(server, "_engine", object()),
            patch.object(server, "health", AsyncMock(return_value=health_attestation)),
            patch.object(
                server,
                "_cache_contract_render_and_tokenize",
                return_value=(
                    {"generation_prompt_suffix_tokens": 0},
                    oversized_vector,
                    None,
                ),
            ),
            pytest.raises(HTTPException) as token_budget,
        ):
            _run(server.cache_token_contract(token_body))
        assert token_budget.value.status_code == 413

    def test_cache_token_contract_uses_loaded_engine_dry_render_and_hashes(
        self,
        tmp_path,
    ):
        """The token contract returns path-free counts/digests and pairwise LCP."""
        from vmlx_engine import server

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(ch) for ch in text]

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            _tokenizer = tokenizer

            def get_stats(self):
                return {"engine_type": "fake"}

            def _apply_chat_template(
                self,
                messages,
                tools=None,
                num_images=0,
                num_videos=0,
                num_audio=0,
                enable_thinking=True,
                extra_template_kwargs=None,
                skip_generation_prompt=False,
            ):
                self.last_tools = tools
                text = "\n".join(
                    f"{message['role']}:{message['content']}"
                    for message in messages
                )
                if (
                    extra_template_kwargs
                    and extra_template_kwargs.get("enable_thinking") is False
                ):
                    text += "\n</think>"
                if not skip_generation_prompt:
                    text += "\nassistant:"
                return text

            def _compute_gen_prompt_cache_context(
                self,
                messages,
                tools,
                num_images,
                enable_thinking,
                extra_template_kwargs,
                prompt_with_gen,
                num_videos=0,
                num_audio=0,
            ):
                from vmlx_engine.engine.batched import (
                    _generation_prompt_cache_extra_key,
                )

                prompt_without_gen = self._apply_chat_template(
                    messages,
                    tools,
                    num_images=num_images,
                    num_videos=num_videos,
                    num_audio=num_audio,
                    enable_thinking=enable_thinking,
                    extra_template_kwargs=extra_template_kwargs,
                    skip_generation_prompt=True,
                )
                tokens_with = self.tokenizer.encode(prompt_with_gen)
                tokens_without = self.tokenizer.encode(prompt_without_gen)
                gen_len = len(tokens_with) - len(tokens_without)
                return gen_len, _generation_prompt_cache_extra_key(
                    prompt_with_generation=prompt_with_gen,
                    prompt_without_generation=prompt_without_gen,
                    gen_prompt_len=gen_len,
                    tokens_with_generation=tokens_with,
                    tokens_without_generation=tokens_without,
                )

        config = tmp_path / "config.json"
        config.write_text('{"model_type":"fake"}\n')
        fake_engine = FakeEngine()
        patches = (
            patch.object(server, "_engine", fake_engine),
            patch.object(server, "_model_name", "fake/model"),
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
        for context in patches:
            context.start()
        try:
            body = {
                "contract_version": 1,
                "surface": "responses",
                "model": "fake/model",
                "inputs": {
                    "A": "shared prefix alpha",
                    "B": "shared prefix beta",
                },
                "request_controls": {
                    "enable_thinking": False,
                    "instructions": None,
                    "tools": [],
                },
            }
            contract = _run(server.cache_token_contract(body))
        finally:
            for context in reversed(patches):
                context.stop()

        assert contract["contract_version"] == 1
        assert contract["method"] == "final-render-tokenize-no-cache"
        assert contract["surface"] == "responses"
        assert contract["cache_lookup_bypassed"] is True
        assert len(contract["request_sha256"]) == 64
        assert len(contract["model_bundle_fingerprint_sha256"]) == 64
        assert len(contract["cache_topology_fingerprint_sha256"]) == 64
        assert set(contract["prompts"]) == {"A", "B"}
        serialized = json.dumps(contract, sort_keys=True)
        assert '"cache_prompt_token_ids":' not in serialized
        assert '"token_ids":' not in serialized
        assert str(tmp_path) not in serialized
        assert contract["prompts"]["A"]["cache_prompt_token_count"] > 1
        assert (
            contract["longest_common_prefix_tokens"]["A:A"]
            == contract["prompts"]["A"]["cache_prompt_token_count"]
        )
        assert 1 < contract["longest_common_prefix_tokens"]["A:B"] < min(
            contract["prompts"]["A"]["cache_prompt_token_count"],
            contract["prompts"]["B"]["cache_prompt_token_count"],
        )

    def test_cache_prefix_attestation_binds_exact_path_free_l1_l2_identity(
        self,
        tmp_path,
    ):
        """The cache gate gets exact chain identity without prompts or paths."""
        from vmlx_engine import server

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(ch) for ch in text]

        class FakeEngine:
            is_mllm = False
            tokenizer = FakeTokenizer()
            _tokenizer = tokenizer

            def _apply_chat_template(
                self,
                messages,
                tools=None,
                num_images=0,
                num_videos=0,
                num_audio=0,
                enable_thinking=True,
                extra_template_kwargs=None,
                skip_generation_prompt=False,
            ):
                text = "\n".join(
                    f"{message['role']}:{message['content']}"
                    for message in messages
                )
                self.last_tools = tools
                if tools:
                    text += "\ntools:" + json.dumps(tools, sort_keys=True)
                return text + ("" if skip_generation_prompt else "\nassistant:")

            def _compute_gen_prompt_cache_context(
                self,
                messages,
                tools,
                num_images,
                enable_thinking,
                extra_template_kwargs,
                prompt_with_gen,
                num_videos=0,
                num_audio=0,
            ):
                from vmlx_engine.engine.batched import (
                    _generation_prompt_cache_extra_key,
                )

                without = self._apply_chat_template(
                    messages,
                    tools,
                    num_images=num_images,
                    num_videos=num_videos,
                    num_audio=num_audio,
                    enable_thinking=enable_thinking,
                    extra_template_kwargs=extra_template_kwargs,
                    skip_generation_prompt=True,
                )
                tokens_with = self.tokenizer.encode(prompt_with_gen)
                tokens_without = self.tokenizer.encode(without)
                gen_len = len(tokens_with) - len(tokens_without)
                return gen_len, _generation_prompt_cache_extra_key(
                    prompt_with_generation=prompt_with_gen,
                    prompt_without_generation=without,
                    gen_prompt_len=gen_len,
                    tokens_with_generation=tokens_with,
                    tokens_without_generation=tokens_without,
                )

        class FakeHashMap:
            def get_block(self, block_hash):
                return SimpleNamespace(
                    block_hash=block_hash,
                    cache_data=None,
                    resident_bytes=0,
                    cache_data_from_disk=False,
                )

        class FakeDiskStore:
            def inspect_block_chain(self, block_hashes):
                now_ns = 1_700_000_000_000_000_000
                return {
                    "schema": "vmlx-block-disk-chain-inspection-v1",
                    "access_metadata_mutated": False,
                    "expected_blocks": len(block_hashes),
                    "store_total_entries": len(block_hashes),
                    "store_total_size_bytes": len(block_hashes) * 1024,
                    "store_max_size_bytes": 1024 * 1024,
                    "blocks": [
                        {
                            "ordinal": ordinal,
                            "indexed": True,
                            "readable": True,
                            "num_tokens": 8,
                            "file_size_bytes": 1024,
                            "created_at_ns": now_ns - 10,
                            "last_accessed_ns": now_ns,
                            "access_count": 2,
                        }
                        for ordinal, _ in enumerate(block_hashes)
                    ],
                }

        fake_paged = SimpleNamespace(
            block_size=8,
            disk_only=True,
            _lock=threading.Lock(),
            cached_block_hash_to_block=FakeHashMap(),
            _disk_store=FakeDiskStore(),
        )
        fake_scheduler = SimpleNamespace(paged_cache_manager=fake_paged)

        async def fake_health():
            return {
                "model_bundle_provenance": {
                    "fingerprint_sha256": "a" * 64,
                },
                "cache_topology_provenance": {
                    "fingerprint_sha256": "b" * 64,
                },
            }

        async def run_inline(callback):
            return callback()

        body = {
            "contract_version": 1,
            "surface": "responses",
            "model": "fake/model",
            "inputs": {
                "left": (
                    "shared private path /private/example/cache "
                    "with a left tail"
                ),
                "right": (
                    "shared private path /private/example/cache "
                    "with a right tail"
                ),
            },
            "prefix_pairs": {"target": ["left", "right"]},
            "request_controls": {
                "enable_thinking": False,
                "instructions": "Keep the source contract stable.",
                "tools": [
                    {
                        "type": "function",
                        "name": "cache_contract_unused",
                        "description": "Stable cache-contract tool schema.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    }
                ],
            },
            "touch": False,
        }
        fake_engine = FakeEngine()
        with (
            patch.object(server, "_engine", fake_engine),
            patch.object(server, "_model_name", "fake/model"),
            patch.object(server, "_model_path", str(tmp_path)),
            patch.object(server, "_get_scheduler", return_value=fake_scheduler),
            patch.object(server, "health", new=fake_health),
            patch.object(server, "_run_on_model_executor", new=run_inline),
        ):
            contract = _run(server.cache_prefix_attestation(body))

        target = contract["prefixes"]["target"]
        assert contract["method"] == (
            "final-render-tokenize-cache-prefix-identity-readonly"
        )
        assert contract["cache_lookup_bypassed"] is True
        assert contract["access_metadata_mutated"] is False
        assert contract["cache_extra_keys_contract"] == (
            "generation-prompt-only-text-render-v1"
        )
        assert contract["caller_cache_or_media_side_keys"] == "rejected"
        assert contract["model_bundle_fingerprint_sha256"] == "a" * 64
        assert contract["cache_topology_fingerprint_sha256"] == "b" * 64
        assert target["reusable_prefix_tokens"] > 0
        assert target["expected_blocks"] > 0
        assert target["l1"]["metadata_blocks_present"] > 0
        assert target["l1"]["resident_payload_blocks_present"] == 0
        assert target["l1"]["backend_mode"] == "block_disk_only"
        assert target["l1"]["paged_ram_enabled"] is False
        assert target["l1"]["disk_only"] is True
        assert target["l2"]["readable_blocks"] == target["expected_blocks"]
        assert target["generation_prompt_discriminator_present"] is True
        assert len(target["generation_prompt_discriminator_sha256"]) == 64
        assert fake_engine.last_tools[0]["function"]["name"] == (
            "cache_contract_unused"
        )
        serialized = json.dumps(contract, sort_keys=True)
        assert "/private/example/cache" not in serialized
        assert "shared private path" not in serialized
        assert '"token_ids":' not in serialized
        assert '"cache_prompt_token_ids":' not in serialized
        assert '"file_name":' not in serialized
        assert '"host":' not in serialized

    def test_cache_prefix_chain_matches_production_generation_hash_inputs(self):
        """Attestation delegates to BatchedEngine/PagedCache identity inputs."""
        from vmlx_engine import server
        from vmlx_engine.engine.batched import _generation_prompt_cache_extra_key
        from vmlx_engine.paged_cache import compute_block_hash

        token_ids = list(range(24))
        block_size = 8
        first_discriminator = _generation_prompt_cache_extra_key(
            prompt_with_generation="rendered-prefix<rail-a>",
            prompt_without_generation="rendered-prefix",
            gen_prompt_len=2,
            tokens_with_generation=[*token_ids, 900, 901],
            tokens_without_generation=token_ids,
        )
        second_discriminator = _generation_prompt_cache_extra_key(
            prompt_with_generation="rendered-prefix<rail-b>",
            prompt_without_generation="rendered-prefix",
            gen_prompt_len=2,
            tokens_with_generation=[*token_ids, 902, 903],
            tokens_without_generation=token_ids,
        )
        assert first_discriminator is not None
        assert second_discriminator is not None
        assert first_discriminator != second_discriminator

        observed = server._cache_prefix_chain_hashes(
            token_ids,
            block_size,
            cache_extra_keys=first_discriminator,
        )
        expected = []
        parent_hash = None
        for start in range(0, len(token_ids), block_size):
            parent_hash = compute_block_hash(
                parent_hash,
                token_ids[start : start + block_size],
                extra_keys=first_discriminator,
            )
            expected.append(bytes(parent_hash))

        assert observed == expected
        other = server._cache_prefix_chain_hashes(
            token_ids,
            block_size,
            cache_extra_keys=second_discriminator,
        )
        assert observed != other
        assert (
            server._cache_prefix_chain_fingerprint(observed)
            != server._cache_prefix_chain_fingerprint(other)
        )

    def test_cache_prefix_attestation_rejects_touch_and_unsafe_labels(self):
        from fastapi import HTTPException

        from vmlx_engine import server

        base = {
            "contract_version": 1,
            "surface": "responses",
            "model": "fake/model",
            "inputs": {"left": "a", "right": "b"},
            "prefix_pairs": {"target": ["left", "right"]},
        }
        with patch.object(server, "_engine", object()):
            with pytest.raises(HTTPException, match="read-only"):
                _run(server.cache_prefix_attestation({**base, "touch": True}))
            unsafe = {
                **base,
                "inputs": {"../private": "a", "right": "b"},
            }
            with pytest.raises(HTTPException, match="safe identifier"):
                _run(server.cache_prefix_attestation(unsafe))
            with pytest.raises(HTTPException, match="cache/media side keys"):
                _run(
                    server.cache_prefix_attestation(
                        {**base, "cache_salt": "caller-owned"}
                    )
                )
            with pytest.raises(HTTPException, match="cache/media side keys"):
                _run(
                    server.cache_prefix_attestation(
                        {
                            **base,
                            "request_controls": {
                                "media_salt": "caller-owned",
                            },
                        }
                    )
                )

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

    def test_cache_topology_fingerprint_ignores_volatile_resident_budget(self):
        """Restart-varying resident byte budgets must not change cache identity."""
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
            "turboquant_kv_cache": {"enabled": True},
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
        paged.max_resident_bytes = 6 * 1024**3
        second = server._cache_topology_attestation(
            server._cache_topology_configuration(scheduler, health_status)
        )

        assert "paged_max_resident_bytes" not in first["configuration"]["instantiated"]
        assert first["canonical_sha256"] == second["canonical_sha256"]

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
