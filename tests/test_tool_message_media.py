"""Media inside `tool` messages reaches the model with a placeholder (audit 2026-10-07, ISSUES I-27).

Agent screenshot tools return an image in the tool result. Before: the batched engine never collected it (Flash-Next
4S answered about a "black screen" for a solid blue screenshot, HTTP 200) and the SimpleEngine path collected it with
no placeholder (27B: HTTP 400 "Image features and image tokens do not match: tokens: 0, features 64"; with a video in
the chat: "Cannot align media items with prompt order"). The Qwen templates render tool content through
render_content() inside <tool_response>, so a marker list is the native form.
"""
from vmlx_engine.api.utils import extract_multimodal_content
from vmlx_engine.models.mllm import MLXMultimodalLM, _order_images_and_video_frames

MSGS = [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "U1"}}, {"type": "text", "text": "a"}]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "screenshot",
     "content": [{"type": "image_url", "image_url": {"url": "T1"}}, {"type": "text", "text": "shot"}]},
    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "V1"}},
                                 {"type": "input_image", "image_url": "U2"}, {"type": "text", "text": "b"}]},
]


def test_collector_keeps_tool_media_in_request_order():
    _, images, videos = extract_multimodal_content(MSGS)
    assert images == ["U1", "T1", "U2"]
    assert videos == ["V1"]


def test_processor_messages_give_tool_media_a_placeholder_and_keep_tool_ids():
    lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    chat, images, videos, _ = lm._extract_multimodal_messages(MSGS)
    tool = next(m for m in chat if m["role"] == "tool")
    assert [p["type"] for p in tool["content"]] == ["image", "text"]
    assert tool["tool_call_id"] == "c1" and tool["name"] == "screenshot"
    assert images == ["U1", "T1", "U2"]
    assert _order_images_and_video_frames(chat, images, [["F1", "F2"]]) == ["U1", "T1", "F1", "F2", "U2"]


def test_text_only_tool_messages_stay_strings():
    lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    msgs = MSGS[:2] + [{"role": "tool", "tool_call_id": "c1", "content": "42"}]
    chat, _, _, _ = lm._extract_multimodal_messages(msgs)
    assert next(m for m in chat if m["role"] == "tool")["content"] == "42"
