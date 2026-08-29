"""Session-scoped adaptive-MTP profile contract.

The per-request controller previously restarted every request at the
configured depth and re-ran the whole AR-vs-MTP experiment (48-64 cycles)
before demoting, so short/first turns paid for a lesson no later turn
remembered. These tests pin the profile store that fixes that lifetime.
"""

from vmlx_engine.native_mtp_profile import (
    REPROBE_EVERY_REQUESTS,
    NativeMTPProfileStore,
    profile_key,
)


class TestProfileKey:
    def test_sampler_class_split(self):
        greedy = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=100)
        sampled = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=100)
        assert greedy[0] == "greedy"
        assert sampled[0] == "sampled"
        assert greedy != sampled

    def test_restored_prefix_split(self):
        cold = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=100)
        warm = profile_key(temperature=0.0, restored_prefix=True, prompt_tokens=100)
        assert cold != warm

    def test_context_buckets(self):
        short = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        medium = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=8000)
        long = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=40000)
        assert short[2] == "short"
        assert medium[2] == "medium"
        assert long[2] == "long"


class TestStartDepth:
    def test_unseen_profile_starts_at_d1_never_configured_d3(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 1, (
            "An unseen profile must start at the bounded near-AR D1 probe, "
            "never optimistically at the configured D2/D3 for a whole request."
        )
        assert source == "unseen_probe_d1"

    def test_learned_depth_is_reused(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=2, fallback_to_ar=False,
                      fallback_reason=None, finish_reason="stop")
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 2
        assert source == "profile_learned_d2"

    def test_learned_depth_clamped_to_configured_ceiling(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=3, fallback_to_ar=False,
                      fallback_reason=None, finish_reason="stop")
        depth, _ = store.start_depth(key, configured_depth=2)
        assert depth == 2

    def test_ar_learned_profile_skips_activation(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost_ratio=1.2", finish_reason="fallback_to_ar")
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 0
        assert source == "profile_ar_skip"

    def test_ar_profile_reprobes_on_bounded_interval(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        sources = [store.start_depth(key, configured_depth=3)
                   for _ in range(REPROBE_EVERY_REQUESTS)]
        skips = [s for s in sources if s[0] == 0]
        probes = [s for s in sources if s[0] == 1]
        assert len(probes) == 1, "exactly one bounded re-probe per interval"
        assert probes[0][1] == "ar_reprobe_d1"
        assert len(skips) == REPROBE_EVERY_REQUESTS - 1

    def test_keys_do_not_poison_each_other(self):
        store = NativeMTPProfileStore()
        sampled = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        greedy = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        store.observe(sampled, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        depth, source = store.start_depth(greedy, configured_depth=3)
        assert depth == 1
        assert source == "unseen_probe_d1"


class TestObserve:
    def test_error_finishes_prove_nothing(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=3, fallback_to_ar=False,
                      fallback_reason=None, finish_reason="error")
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 1
        assert source == "unseen_probe_d1"

    def test_recovery_after_ar_reprobe_succeeds(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.7, restored_prefix=False, prompt_tokens=64)
        store.observe(key, final_depth=1, fallback_to_ar=True,
                      fallback_reason="cost", finish_reason="fallback_to_ar")
        # a later successful probe re-learns a real depth
        store.observe(key, final_depth=2, fallback_to_ar=False,
                      fallback_reason=None, finish_reason="stop")
        depth, source = store.start_depth(key, configured_depth=3)
        assert depth == 2
        assert source == "profile_learned_d2"

    def test_snapshot_shape(self):
        store = NativeMTPProfileStore()
        key = profile_key(temperature=0.0, restored_prefix=True, prompt_tokens=64)
        store.observe(key, final_depth=2, fallback_to_ar=False,
                      fallback_reason=None, finish_reason="stop",
                      values_tok_s={"d1": 30.0, "d2": 41.5, "d3": None})
        snap = store.snapshot()
        assert "greedy|restored|short" in snap
        entry = snap["greedy|restored|short"]
        assert entry["learned_depth"] == 2
        assert entry["requests_observed"] == 1
        assert entry["last_values_tok_s"]["d2"] == 41.5


class TestSeedPathIntegration:
    def _build_generator(self, monkeypatch):
        import mlx.core as mx

        from vmlx_engine.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        monkeypatch.delenv("VMLX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.delenv("VMLINUX_NATIVE_MTP_DEPTH", raising=False)
        monkeypatch.setenv("VMLINUX_NATIVE_MTP_USE_TUNING", "0")
        vocab_size = 16

        def logits_for(targets):
            rows = []
            for target in targets:
                row = [-100.0] * vocab_size
                row[int(target)] = 100.0
                rows.append(row)
            return mx.array([rows], dtype=mx.float32)

        class _LM:
            def make_mtp_cache(self):
                return []

            def __call__(self, input_ids, cache=None, return_hidden=False, **_kw):
                tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
                logits = logits_for([(tokens[-1] + 1) % vocab_size])
                if return_hidden:
                    return logits, mx.ones((1, 1, 4), dtype=mx.float32)
                return logits

            def mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                            return_hidden=False):
                next_id = int(next_token_ids.reshape(-1).tolist()[-1])
                logits = logits_for([(next_id + 1) % vocab_size])
                if return_hidden:
                    return logits, mx.ones((1, 1, 4), dtype=mx.float32)
                return logits

        lm = _LM()
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.language_model = lm
        generator.model = type("_VLM", (), {"language_model": lm})()
        generator.stop_tokens = set()
        generator._stats = MLLMBatchStats()
        generator._native_mtp_enabled_for_request = lambda _req: True

        def greedy_sampler(logits):
            return mx.argmax(logits, axis=-1).astype(mx.uint32)

        greedy_sampler._vmlx_accepts_logits = True
        generator._make_request_sampler = lambda _req: greedy_sampler

        req = MLLMBatchRequest(
            uid=0, request_id="profile-seed-test", prompt="",
            max_tokens=8, temperature=0.0,
        )
        req.input_ids = mx.array([101, 102])
        req._original_token_ids = [101, 102]
        first_token = mx.array([2], dtype=mx.uint32)
        return generator, req, first_token

    def test_unseen_profile_seeds_depth1_and_ar_profile_skips(self, monkeypatch):
        import mlx.core as mx  # noqa: F401 — fixture requires MLX

        generator, req, first_token = self._build_generator(monkeypatch)

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.depth == 1, (
            "unseen profile must seed the bounded D1 probe, not configured D3"
        )
        assert state.stats.profile_seed == "unseen_probe_d1"
        assert state.profile_key is not None

        # Teach the profile that AR wins, then reseed: activation must skip.
        generator._native_mtp_profiles.observe(
            state.profile_key, final_depth=1, fallback_to_ar=True,
            fallback_reason="cost", finish_reason="fallback_to_ar",
        )
        delattr(req, "_native_mtp_state")
        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is False, "AR-learned profile must skip MTP activation entirely"

    def test_explicit_env_depth_override_bypasses_profile(self, monkeypatch):
        generator, req, first_token = self._build_generator(monkeypatch)
        monkeypatch.setenv("VMLX_NATIVE_MTP_DEPTH", "3")

        assert generator._seed_native_mtp_from_prefill(
            req, [object()], first_token, [None]
        ) is True
        state = req._native_mtp_state
        assert state.depth == 3, (
            "an explicit user depth override must not be second-guessed by "
            "the session profile"
        )
        assert state.stats.profile_seed == "configured"


class TestStateIntegration:
    def test_state_carries_profile_key_and_stats_report_seed(self):
        from vmlx_engine.mllm_batch_generator import (
            MLLMNativeMTPState,
            MLLMNativeMTPStats,
        )

        state = MLLMNativeMTPState(profile_key=("greedy", False, "short"))
        assert state.profile_key == ("greedy", False, "short")
        stats = MLLMNativeMTPStats()
        stats.profile_seed = "profile_learned_d2"
        payload = stats.to_dict(
            request_id="r1", finish_reason="stop", final_depth=2
        )
        assert payload["profile_seed"] == "profile_learned_d2"
