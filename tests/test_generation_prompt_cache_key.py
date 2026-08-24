from vmlx_engine.engine.batched import _generation_prompt_cache_extra_key


def test_generation_prompt_cache_key_separates_thinking_on_off_suffixes():
    prefix = "<user>hi</user><assistant>"
    on_prompt = prefix + "<think>"
    off_prompt = prefix + "</think>"

    on_key = _generation_prompt_cache_extra_key(
        prompt_with_generation=on_prompt,
        prompt_without_generation=prefix,
        gen_prompt_len=1,
        tokens_with_generation=[10, 20, 30],
        tokens_without_generation=[10, 20],
    )
    off_key = _generation_prompt_cache_extra_key(
        prompt_with_generation=off_prompt,
        prompt_without_generation=prefix,
        gen_prompt_len=1,
        tokens_with_generation=[10, 20, 31],
        tokens_without_generation=[10, 20],
    )

    assert on_key is not None
    assert off_key is not None
    assert on_key["generation_prompt"].startswith("v1:1:")
    assert off_key["generation_prompt"].startswith("v1:1:")
    assert on_key != off_key


def test_generation_prompt_cache_key_is_stable_for_identical_suffix():
    prefix = "<user>hi</user><assistant>"
    prompt = prefix + "<think>"

    first = _generation_prompt_cache_extra_key(
        prompt_with_generation=prompt,
        prompt_without_generation=prefix,
        gen_prompt_len=1,
        tokens_with_generation=[10, 20, 30],
        tokens_without_generation=[10, 20],
    )
    second = _generation_prompt_cache_extra_key(
        prompt_with_generation=prompt,
        prompt_without_generation=prefix,
        gen_prompt_len=1,
        tokens_with_generation=[10, 20, 30],
        tokens_without_generation=[10, 20],
    )

    assert first == second


def test_generation_prompt_cache_key_ignores_user_tail_for_replacing_templates():
    """A replacement-style generation prompt must not salt the whole prompt.

    MiniMax M2.x thinking-off rendering is one real example: the engine adds
    an empty think sentinel to both renders, while ``add_generation_prompt``
    inserts the assistant marker before it.  The two strings are therefore not
    literal prefix extensions even though the generation suffix is identical.
    """

    first = _generation_prompt_cache_extra_key(
        prompt_with_generation=(
            "<user>shared prefix tail A</user>"
            "<assistant><think></think>"
        ),
        prompt_without_generation=(
            "<user>shared prefix tail A</user><think></think>"
        ),
        gen_prompt_len=3,
        tokens_with_generation=[10, 11, 20, 30, 31, 32],
        tokens_without_generation=[10, 11, 20],
    )
    second = _generation_prompt_cache_extra_key(
        prompt_with_generation=(
            "<user>shared prefix tail B differs</user>"
            "<assistant><think></think>"
        ),
        prompt_without_generation=(
            "<user>shared prefix tail B differs</user><think></think>"
        ),
        gen_prompt_len=3,
        tokens_with_generation=[40, 41, 42, 30, 31, 32],
        tokens_without_generation=[40, 41, 42],
    )

    assert first == second


def test_generation_prompt_cache_key_skips_when_nothing_was_stripped():
    assert (
        _generation_prompt_cache_extra_key(
            prompt_with_generation="<user>hi</user>",
            prompt_without_generation="<user>hi</user>",
            gen_prompt_len=0,
            tokens_with_generation=[1, 2],
            tokens_without_generation=[1, 2],
        )
        is None
    )


def test_generation_prompt_cache_context_returns_tuple_when_tokenizer_missing():
    from types import SimpleNamespace

    from vmlx_engine.engine.batched import BatchedEngine

    engine = object.__new__(BatchedEngine)
    engine._tokenizer = None
    engine._processor = None
    engine._apply_chat_template = lambda *args, **kwargs: "<user>hi</user>"

    assert engine._compute_gen_prompt_cache_context(
        messages=[],
        tools=None,
        num_images=0,
        enable_thinking=True,
        extra_template_kwargs=None,
        prompt_with_gen="<user>hi</user><assistant><think>",
    ) == (0, None)

    engine._processor = SimpleNamespace()

    assert engine._compute_gen_prompt_cache_context(
        messages=[],
        tools=None,
        num_images=0,
        enable_thinking=True,
        extra_template_kwargs=None,
        prompt_with_gen="<user>hi</user><assistant><think>",
    ) == (0, None)


def test_generation_prompt_cache_context_dsv4_rails_share_constant_key():
    """DSV4 <think>/</think> rails must share one prefix-cache namespace.

    DSV4 truncates stored prefix KV exactly to the stripped-key boundary and
    replays the request's own generation-prompt suffix on fetch, so a prefix
    stored under the thinking rail is replay-exact for the chat rail. Hashing
    the suffix into block keys forced the visible-answer pass to re-prefill
    the entire prompt (59s stall at 15k tokens).
    """
    from types import SimpleNamespace

    from vmlx_engine.engine.batched import BatchedEngine

    base = "<user>hi</user><assistant>"

    def make_engine(family):
        engine = object.__new__(BatchedEngine)
        engine._processor = None
        engine._tokenizer = SimpleNamespace(
            encode=lambda prompt, add_special_tokens=False: list(
                prompt.encode("utf-8")
            )
        )
        engine._apply_chat_template = lambda *args, **kwargs: base
        engine._model_family_name = lambda: family
        return engine

    def context_for(engine, suffix):
        return engine._compute_gen_prompt_cache_context(
            messages=[],
            tools=None,
            num_images=0,
            enable_thinking=True,
            extra_template_kwargs=None,
            prompt_with_gen=base + suffix,
        )

    dsv4 = make_engine("deepseek_v4")
    think_len, think_key = context_for(dsv4, "<think>")
    chat_len, chat_key = context_for(dsv4, "</think>")

    assert think_len > 0
    assert chat_len > 0
    assert think_key == {"generation_prompt": "dsv4-replay-exact"}
    assert chat_key == think_key

    other = make_engine("qwen3")
    _, on_key = context_for(other, "<think>")
    _, off_key = context_for(other, "</think>")
    assert on_key != off_key
    assert on_key != think_key


def test_mllm_media_cache_side_key_merge_preserves_generation_prompt_axis():
    from vmlx_engine.cache_key import CACHE_EXTRA_SCOPES_KEY
    from vmlx_engine.mllm_batch_generator import _merge_mllm_cache_extra_keys

    assert _merge_mllm_cache_extra_keys(
        {"generation_prompt": "v1:1:abc"},
        {"mllm_media": "def"},
    ) == {
        "generation_prompt": "v1:1:abc",
        "mllm_media": "def",
    }

    assert _merge_mllm_cache_extra_keys(
        {
            "generation_prompt": "v1:1:abc",
            CACHE_EXTRA_SCOPES_KEY: {"generation_prompt": 2},
        },
        {
            "mllm_media_0000": "def",
            CACHE_EXTRA_SCOPES_KEY: {"mllm_media_0000": 6},
        },
    ) == {
        "generation_prompt": "v1:1:abc",
        "mllm_media_0000": "def",
        CACHE_EXTRA_SCOPES_KEY: {
            "generation_prompt": 2,
            "mllm_media_0000": 6,
        },
    }


def test_mllm_media_cache_scopes_each_item_at_its_own_placeholder():
    from types import SimpleNamespace

    from vmlx_engine.cache_key import (
        CACHE_EXTRA_SCOPES_KEY,
        cache_extra_keys_for_token_range,
    )
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    generator = object.__new__(MLLMBatchGenerator)
    generator.model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    generator.language_model = SimpleNamespace(config=None)

    def request(second: str):
        return SimpleNamespace(
            images=["image-a", second],
            videos=None,
            audio=None,
            audios=None,
            image_token_budget=None,
            image_grid_thw=None,
            video_grid_thw=None,
            audio_codes=None,
            audio_embeds=None,
            audio_features=None,
            audio_features_mask=None,
            pixel_values=None,
            pixel_values_videos=None,
            video_pixel_values=None,
        )

    tokens = [1, 2, 99, 99, 3, 4, 99, 99, 5]
    first_request = request("image-b")
    changed_request = request("image-c")
    first = generator._media_scoped_cache_extra_keys(first_request, tokens)
    changed = generator._media_scoped_cache_extra_keys(changed_request, tokens)

    assert first_request._media_cache_scope == {
        "mode": "per_media_placeholder",
        "items": 2,
        "modalities": ["image", "image"],
        "boundaries": [2, 6],
    }
    assert first[CACHE_EXTRA_SCOPES_KEY] == {
        "mllm_media_0000": 2,
        "mllm_media_0001": 6,
    }
    assert cache_extra_keys_for_token_range(first, 0, 2) is None
    first_media = cache_extra_keys_for_token_range(first, 2, 6)
    changed_first_media = cache_extra_keys_for_token_range(changed, 2, 6)
    assert first_media == changed_first_media
    assert set(first_media) == {"mllm_media_0000", CACHE_EXTRA_SCOPES_KEY}
    assert cache_extra_keys_for_token_range(first, 6, 8) != (
        cache_extra_keys_for_token_range(changed, 6, 8)
    )


def test_mllm_media_cache_maps_video_image_audio_in_causal_token_order():
    from types import SimpleNamespace

    from vmlx_engine.cache_key import CACHE_EXTRA_SCOPES_KEY
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    generator = object.__new__(MLLMBatchGenerator)
    generator.model = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=99,
            video_token_id=88,
            audio_token_id=77,
        )
    )
    generator.language_model = SimpleNamespace(config=None)
    request = SimpleNamespace(
        images=["image-a"],
        videos=["video-a"],
        audio=["audio-a"],
        audios=None,
        image_token_budget=None,
        video_fps=2.0,
        video_max_frames=8,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_codes=None,
        audio_embeds=None,
        audio_features=None,
        audio_features_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        video_pixel_values=None,
    )

    scoped = generator._media_scoped_cache_extra_keys(
        request,
        [1, 88, 88, 2, 99, 99, 3, 77, 77, 4],
    )

    assert request._media_cache_scope == {
        "mode": "per_media_placeholder",
        "items": 3,
        "modalities": ["video", "image", "audio"],
        "boundaries": [1, 4, 7],
    }
    assert scoped[CACHE_EXTRA_SCOPES_KEY] == {
        "mllm_media_0000": 1,
        "mllm_media_0001": 4,
        "mllm_media_0002": 7,
    }


def test_step_media_cache_groups_adaptive_patch_runs_per_source_item():
    """Step emits N crop runs + one full-image run for each source image."""
    from types import SimpleNamespace

    from vmlx_engine.cache_key import (
        CACHE_EXTRA_SCOPES_KEY,
        cache_extra_keys_for_token_range,
    )
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    generator = object.__new__(MLLMBatchGenerator)
    generator._model_type = "step3p7"
    generator.model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    generator.language_model = SimpleNamespace(config=None)

    def request(second: str):
        return SimpleNamespace(
            images=["image-a", second],
            videos=None,
            audio=None,
            audios=None,
            image_token_budget=None,
            image_grid_thw=None,
            video_grid_thw=None,
            audio_codes=None,
            audio_embeds=None,
            audio_features=None,
            audio_features_mask=None,
            pixel_values=None,
            pixel_values_videos=None,
            video_pixel_values=None,
            extra_kwargs={"num_patches": [2, 1]},
        )

    # First image owns three runs (two crops + full image); second owns two.
    tokens = [1, 2, 99, 99, 3, 99, 4, 99, 99, 5, 6, 99, 99, 7, 99, 99, 8]
    first_request = request("image-b")
    changed_request = request("image-c")
    first = generator._media_scoped_cache_extra_keys(first_request, tokens)
    changed = generator._media_scoped_cache_extra_keys(changed_request, tokens)

    assert first_request._media_cache_scope == {
        "mode": "per_media_placeholder",
        "items": 2,
        "modalities": ["image", "image"],
        "boundaries": [2, 11],
        "placeholder_runs": 5,
        "run_group_sizes": [3, 2],
    }
    assert first[CACHE_EXTRA_SCOPES_KEY] == {
        "mllm_media_0000": 2,
        "mllm_media_0001": 11,
    }
    assert cache_extra_keys_for_token_range(first, 2, 11) == (
        cache_extra_keys_for_token_range(changed, 2, 11)
    )
    assert cache_extra_keys_for_token_range(first, 11, len(tokens)) != (
        cache_extra_keys_for_token_range(changed, 11, len(tokens))
    )


def test_paged_cache_schema_version_tracks_tool_and_rotating_contracts():
    from vmlx_engine.prefix_cache import PAGED_CACHE_SCHEMA_VERSION

    assert "v12" in PAGED_CACHE_SCHEMA_VERSION
    assert "qwen_tool" in PAGED_CACHE_SCHEMA_VERSION
    assert "rotating" in PAGED_CACHE_SCHEMA_VERSION
    assert "media_causal_scopes" in PAGED_CACHE_SCHEMA_VERSION


class TestReplayExactFamilyAllowlist:
    """Which families may share prefix KV across reasoning rails.

    A family qualifies only when the generation-prompt strip is ACTIVE for it,
    so stored KV is truncated to the stripped-key boundary and the rail suffix is
    replayed on every fetch. Where the strip is force-disabled (mixed-SWA,
    openpangu) the stored KV still CONTAINS the rail suffix, so sharing would
    restore the wrong causal state.
    """

    def _gen(self, family):
        from vmlx_engine.engine.batched import BatchedEngine

        probe = BatchedEngine.__new__(BatchedEngine)
        probe._model_family_name = lambda: family
        return BatchedEngine._replay_exact_generation_prompt(probe)

    def test_proven_families_share_across_rails(self):
        assert self._gen("deepseek_v4") is True
        assert self._gen("laguna") is True

    def test_strip_disabled_families_must_not_share(self):
        for family in ("gemma4_unified", "muse_glimmer", "openpangu_v2", "minimax_m3"):
            assert self._gen(family) is False, family

    def test_unknown_family_defaults_to_conservative(self):
        assert self._gen("some_new_family") is False
        assert self._gen("") is False
        assert self._gen(None) is False

    def test_env_override_can_force_or_restore(self, monkeypatch):
        monkeypatch.setenv("VMLX_REPLAY_EXACT_GEN_PROMPT", "0")
        # back to DSV4-only, the pre-generalisation behaviour
        assert self._gen("laguna") is False
        assert self._gen("deepseek_v4") is True
        monkeypatch.setenv("VMLX_REPLAY_EXACT_GEN_PROMPT", "1")
        assert self._gen("gemma4_unified") is True
