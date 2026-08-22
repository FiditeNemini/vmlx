# SPDX-License-Identifier: Apache-2.0
"""An image must stay in the message that carried it, however the chat grows.

mlx_vlm's whole-list builder finds the LAST user message and re-attaches EVERY
image to it, stripping them out of the messages that actually carried them
(prompt_utils: extract_text_from_content drops the image part, then
skip_image_token=not is_target puts them all on last_user_idx). For a
single-turn request that is a no-op -- which is exactly why it looked fine and
why every existing probe missed it, since they all used ONE fixed message
list, where the reposition is invisible by construction.

Extend the conversation by one turn and the image MOVES. Measured live on
Qwen3.8-27B VL through /v1/chat/completions:

    request A (4 msgs, image in the last user msg):
        n=2456  media_tokens=72  first_media_at=84
    request B (A + assistant + user, image still in msg 3):
        n=2478  media_tokens=72  first_media_at=2396

Same image, same 72 expanded media tokens, 2312 tokens apart. Every token
after the image shifts, so no cache block hash can match and multimodal prefix
reuse is permanently 0% from the first image turn onward. It presents as a
broken cache while the cache is behaving correctly on a prompt that genuinely
is not the same prompt.

The invariant here is CROSS-REQUEST on purpose: a within-request assertion
cannot see this.
"""

import pytest


def _image_message_indices(built):
    """Indices of built messages whose content carries an image part."""
    out = []
    for index, message in enumerate(built):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image"
            for part in content
        ):
            out.append(index)
    return out


def _build(model_type, messages, num_images):
    """Owner-preserving build, mirroring batched.py's helper."""
    from mlx_vlm.prompt_utils import (
        extract_text_from_content,
        get_message_json,
    )

    def own(content):
        if not isinstance(content, list):
            return 0
        return sum(
            1
            for item in content
            if isinstance(item, dict)
            and item.get("type") in ("image", "image_url", "input_image")
        )

    owned = [own(m.get("content")) for m in messages]
    surplus = max(0, int(num_images) - sum(owned))
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") not in ("system", "assistant", "tool"):
            last_user = index

    built = []
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        if role == "tool" or message.get("tool_calls"):
            built.append(message)
            continue
        text = extract_text_from_content(message.get("content", ""))
        count = owned[index] + (surplus if index == last_user and surplus else 0)
        built.append(
            get_message_json(
                model_type, text, role,
                skip_image_token=(count == 0), num_images=count,
            )
        )
    return built


def _conversation():
    base = [
        {"role": "system", "content": "terse"},
        {"role": "user", "content": "describe"},
        {"role": "assistant", "content": "two words"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what colours"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
    ]
    extended = base + [
        {"role": "assistant", "content": "green red"},
        {"role": "user", "content": "one more"},
    ]
    return base, extended


# The two families on the media prefix-reuse allow-list that are ALSO in
# mlx_vlm's MODEL_CONFIG, so both were nullified by this. (step3p7, the third,
# is absent from MODEL_CONFIG and escaped through the exception fallback.)
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "gemma4"])
def test_image_index_is_stable_under_one_turn_extension(model_type):
    base, extended = _conversation()
    base_idx = _image_message_indices(_build(model_type, base, 1))
    ext_idx = _image_message_indices(_build(model_type, extended, 1))
    assert base_idx == [3], "image not on its own message even in the base case"
    assert ext_idx == base_idx, (
        "image MOVED from message %s to %s when the conversation grew by one "
        "turn -- every token after it shifts and cache reuse dies"
        % (base_idx, ext_idx)
    )


def test_the_upstream_builder_still_has_the_defect_we_route_around():
    """Pin WHY the local builder exists.

    If mlx_vlm ever fixes this, this test fails and the local builder can be
    reconsidered. Without it, a future reader has no way to tell whether the
    extra code is still earning its place.
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    base, extended = _conversation()
    upstream_base = apply_chat_template(
        None, {"model_type": "qwen3_5"}, base,
        num_images=1, return_messages=True,
    )
    upstream_ext = apply_chat_template(
        None, {"model_type": "qwen3_5"}, extended,
        num_images=1, return_messages=True,
    )
    assert _image_message_indices(upstream_base) == [3]
    assert _image_message_indices(upstream_ext) == [5], (
        "upstream mlx_vlm no longer repositions images; the local "
        "owner-preserving builder may be removable"
    )


def test_images_in_two_different_messages_do_not_bunch():
    """The same mechanism misattributes images even on a FIRST request.

    Upstream moves BOTH images onto the last user message, so a model asked
    about "the first image" is shown them in one place with no way to tell
    which turn each came from.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "a"}},
            ],
        },
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "second"},
                {"type": "image_url", "image_url": {"url": "b"}},
            ],
        },
    ]
    assert _image_message_indices(_build("qwen3_5", messages, 2)) == [0, 2]


def test_surplus_images_without_an_owning_part_keep_the_old_convention():
    """Images passed via the images= kwarg have no owning message.

    There is nothing better to key on for those, so they keep landing on the
    last user message -- but they must not disturb an image that DOES have an
    owner.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "a"}},
            ],
        },
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    # 2 images declared, only 1 owned by a message part.
    assert _image_message_indices(_build("qwen3_5", messages, 2)) == [0, 2]


def test_the_engine_uses_the_owner_preserving_builder():
    import inspect

    from vmlx_engine.engine.batched import BatchedEngine

    src = inspect.getsource(BatchedEngine._apply_chat_template)
    assert "_build_messages_preserving_image_owner(" in src
    assert "IMAGES MUST STAY IN THEIR OWN MESSAGE" in src
