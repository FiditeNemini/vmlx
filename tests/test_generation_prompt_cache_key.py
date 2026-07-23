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


def test_mllm_media_cache_side_key_merge_preserves_generation_prompt_axis():
    from vmlx_engine.mllm_batch_generator import _merge_mllm_cache_extra_keys

    assert _merge_mllm_cache_extra_keys(
        {"generation_prompt": "v1:1:abc"},
        {"mllm_media": "def"},
    ) == {
        "generation_prompt": "v1:1:abc",
        "mllm_media": "def",
    }


def test_paged_cache_schema_version_tracks_tool_and_rotating_contracts():
    from vmlx_engine.prefix_cache import PAGED_CACHE_SCHEMA_VERSION

    assert "v11" in PAGED_CACHE_SCHEMA_VERSION
    assert "qwen_tool_continuation" in PAGED_CACHE_SCHEMA_VERSION
    assert "rotating_terminal_window" in PAGED_CACHE_SCHEMA_VERSION
