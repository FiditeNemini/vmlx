"""A literal <think> inside message content is content, never the generation-prompt rail."""
from vmlx_engine.utils.chat_template_kwargs import ensure_thinking_off_sentinel, trailing_open_think_index

USER = "Explain this log line: <think> starts the reasoning rail, then the model writes its plan."


def test_trailing_rail_is_found_only_at_the_end():
    assert trailing_open_think_index("<|im_start|>assistant\n<think>\n") == len("<|im_start|>assistant\n")
    assert trailing_open_think_index(f"<|im_start|>user\n{USER}<|im_end|>\n") == -1
    assert trailing_open_think_index("no tags") == -1


def test_thinking_off_render_without_generation_prompt_keeps_quoted_tag_content():
    prompt = f"<|im_start|>user\n{USER}<|im_end|>\n"            # skip_generation_prompt render
    assert ensure_thinking_off_sentinel(prompt, family_name="qwen3_5") == prompt


def test_thinking_off_still_closes_the_real_trailing_rail():
    prompt = f"<|im_start|>user\n{USER}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    out = ensure_thinking_off_sentinel(prompt, family_name="qwen3_5")
    assert out.startswith(f"<|im_start|>user\n{USER}<|im_end|>\n")    # content untouched
    assert out.endswith("<think>\n</think>\n\n")
