from vmlx_engine.utils.chat_template_kwargs import build_chat_template_kwargs


def test_enable_thinking_sets_thinking_alias():
    kwargs = build_chat_template_kwargs(enable_thinking=False)

    assert kwargs["enable_thinking"] is False
    assert kwargs["thinking"] is False


def test_enable_thinking_wins_over_conflicting_extra_aliases():
    kwargs = build_chat_template_kwargs(
        enable_thinking=False,
        extra={
            "enable_thinking": True,
            "thinking": True,
            "thinking_budget": 2048,
            "tokenize": True,
            "add_generation_prompt": False,
        },
    )

    assert kwargs["enable_thinking"] is False
    assert kwargs["thinking"] is False
    assert kwargs["thinking_budget"] == 2048
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True


def test_processor_path_can_skip_thinking_alias():
    kwargs = build_chat_template_kwargs(
        enable_thinking=True,
        include_thinking_alias=False,
    )

    assert kwargs["enable_thinking"] is True
    assert "thinking" not in kwargs


def test_glm53_preserves_native_history_default_and_explicit_override():
    for family in ("glm5_next", "glm5_next_text"):
        kwargs = build_chat_template_kwargs(enable_thinking=None, model_type=family)
        assert "clear_thinking" not in kwargs
        for choice in (False, True):
            kwargs = build_chat_template_kwargs(
                enable_thinking=None, model_type=family,
                extra={"clear_thinking": choice},
            )
            assert kwargs["clear_thinking"] is choice
    # The unrelated legacy serving policy is unchanged.
    assert build_chat_template_kwargs(
        enable_thinking=None, model_type="glm5"
    )["clear_thinking"] is True
    assert "clear_thinking" not in build_chat_template_kwargs(enable_thinking=None)


def test_model_type_of_reads_dict_and_object_configs():
    from types import SimpleNamespace

    from vmlx_engine.utils.chat_template_kwargs import model_type_of

    assert model_type_of(SimpleNamespace(config={"model_type": "glm5_next"})) == "glm5_next"
    assert model_type_of(SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp"))) == "qwen4_exp"
    assert model_type_of(object()) == ""
