"""Gemma4 tool-parser content-leak fixes (2026-06-22).

Reproduces the live-observed leaks on 2-bit gemma-4-12B tool calls:
- '<audio|>;' leaked via /v1/chat/completions
- 'loader:list_files{path:<|"|>/tmp<|"|>}' leaked via /v1/responses
"""
from vmlx_engine.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


def test_strips_audio_token():
    c = Gemma4ToolParser._clean_special_tokens("The answer<audio|>; is 4")
    assert "<audio|>" not in c
    assert "is 4" in c


def test_strips_gemma_quote_token():
    c = Gemma4ToolParser._clean_special_tokens('name: <|"|>Alice<|"|> done')
    assert '<|"|>' not in c
    assert "Alice" in c and "done" in c


def test_strips_bare_tool_call_residue():
    leaked = 'loader:list_files{path:<|"|>/tmp<|"|>}I cannot do that.'
    c = Gemma4ToolParser._clean_special_tokens(leaked)
    assert "loader:list_files" not in c
    assert '<|"|>' not in c
    assert "I cannot do that." in c


def test_preserves_normal_text():
    txt = "The capital of France is Paris."
    assert Gemma4ToolParser._clean_special_tokens(txt) == txt


def test_preserves_normal_colon_text():
    # must NOT strip legit prose that merely contains a word:word pattern
    txt = "Note: the result was good."
    assert Gemma4ToolParser._clean_special_tokens(txt) == txt
