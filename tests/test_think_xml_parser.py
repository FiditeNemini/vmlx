from vmlx_engine.reasoning.think_xml_parser import ThinkXmlReasoningParser


def test_think_xml_no_tags_with_think_in_prompt_stays_visible_content():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning(
        "The answer is ready.\n\nFINAL=OK"
    )

    assert reasoning is None
    assert content == "The answer is ready.\n\nFINAL=OK"


def test_think_xml_streaming_no_tags_with_think_in_prompt_stays_visible_content():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    delta = parser.extract_reasoning_streaming("", "FINAL=OK", "FINAL=OK")

    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "FINAL=OK"


def test_think_xml_explicit_tags_still_extract_reasoning():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("<think>brief</think>FINAL=OK")

    assert reasoning == "brief"
    assert content == "FINAL=OK"


def test_think_xml_only_end_tag_still_extracts_implicit_reasoning():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("brief</think>FINAL=OK")

    assert reasoning == "brief"
    assert content == "FINAL=OK"


def test_native_whitespace_survives_every_two_chunk_split():
    text = "<think>\nReason.\n</think>\n\nAnswer."
    for split in range(1, len(text)):
        parser = ThinkXmlReasoningParser()
        parser.preserve_native_whitespace = True
        parser.reset_state(think_in_prompt=False)
        previous = ""
        reasoning = content = ""
        for delta in (text[:split], text[split:]):
            current = previous + delta
            result = parser.extract_reasoning_streaming(previous, current, delta)
            if result:
                reasoning += result.reasoning or ""
                content += result.content or ""
            previous = current
        assert reasoning == "\nReason.\n", split
        assert content == "\n\nAnswer.", split
        assert parser.extract_reasoning(text) == (reasoning, content)


def test_native_whitespace_is_selected_only_for_naive_family():
    from types import SimpleNamespace
    from vmlx_engine.server import _new_request_reasoning_parser
    for family, expected in (("naive_n05_flash", True), ("mimo_v2", False)):
        parser = _new_request_reasoning_parser(
            configured_parser=ThinkXmlReasoningParser(),
            model_config=SimpleNamespace(family_name=family),
            effective_think_in_template=False, harmony_active=False,
            stream_surface="chat",
        )
        assert parser.preserve_native_whitespace is expected
