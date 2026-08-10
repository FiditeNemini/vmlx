# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer family resolution.

Muse routes reasoning by RECIPIENT and calls tools in the ATEM dialect, so the
registry must hand out the muse_glimmer reasoning parser and the atem tool
parser — not the generic think-tag/JSON defaults, which would silently mangle
both. Fixture is the real bundle's config.json, trimmed to the identity and
modality fields the registry reads.
"""

import os

from vmlx_engine.model_config_registry import get_model_config_registry

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "muse_glimmer")


def _config():
    return get_model_config_registry().lookup(FIXTURE)


def test_resolves_the_muse_glimmer_family():
    assert _config().family_name == "muse_glimmer"


def test_routes_to_the_atem_and_recipient_parsers():
    cfg = _config()
    assert cfg.tool_parser == "atem"
    assert cfg.reasoning_parser == "muse_glimmer"


def test_declared_multimodal_with_thinking():
    cfg = _config()
    assert cfg.is_mllm is True
    assert cfg.supports_thinking is True
    assert cfg.cache_type == "kv"


def test_eom_is_not_an_eos_token():
    """<|eom|> ends one message with more to follow.

    Treating it as end-of-turn would truncate every reply at the end of the
    reasoning rail, before the answer is emitted.
    """
    cfg = _config()
    assert "<|eom|>" not in (cfg.eos_tokens or [])
    assert "<|eot|>" in cfg.eos_tokens
    assert "<|end_of_text|>" in cfg.eos_tokens


def test_channel_and_media_markers_are_cleaned_from_output():
    cleaned = _config().special_tokens_to_clean or []
    for marker in ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>",
                   "<|patch|>", "<|video|>"):
        assert marker in cleaned, f"{marker} would leak into surfaced text"


def test_parsers_named_here_are_actually_registered():
    """A family may not point at a parser name that does not resolve."""
    from vmlx_engine.reasoning import get_parser
    from vmlx_engine.tool_parsers import ToolParserManager

    cfg = _config()
    assert get_parser(cfg.reasoning_parser).__name__ == "MuseGlimmerReasoningParser"
    assert ToolParserManager.get_tool_parser(cfg.tool_parser).__name__ == "AtemToolParser"
