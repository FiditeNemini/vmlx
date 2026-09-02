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


def test_glm5_defaults_clear_thinking_true_with_caller_override():
    """GLM templates default clear_thinking=false, which replays every prior
    turn's reasoning into the prompt (measured live: ~3k tokens/turn until the
    prefill admission guard rejected the conversation). The kwargs layer pins
    the flat-history default for glm5 families only; an explicit caller value
    and non-GLM families are untouched."""
    from vmlx_engine.utils.chat_template_kwargs import build_chat_template_kwargs

    glm = build_chat_template_kwargs(
        enable_thinking=None, model_type="glm5_next"
    )
    assert glm["clear_thinking"] is True

    glm_text = build_chat_template_kwargs(
        enable_thinking=None, model_type="glm5_next_text"
    )
    assert glm_text["clear_thinking"] is True

    caller = build_chat_template_kwargs(
        enable_thinking=None,
        model_type="glm5_next",
        extra={"clear_thinking": False},
    )
    assert caller["clear_thinking"] is False

    other = build_chat_template_kwargs(
        enable_thinking=None, model_type="qwen4_exp"
    )
    assert "clear_thinking" not in other

    untyped = build_chat_template_kwargs(enable_thinking=None)
    assert "clear_thinking" not in untyped


def test_model_type_of_reads_dict_and_object_configs():
    from types import SimpleNamespace

    from vmlx_engine.utils.chat_template_kwargs import model_type_of

    assert model_type_of(SimpleNamespace(config={"model_type": "glm5_next"})) == "glm5_next"
    assert model_type_of(SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp"))) == "qwen4_exp"
    assert model_type_of(object()) == ""
