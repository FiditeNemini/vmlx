"""Thinking-Off history: keep prior reasoning when the loaded template itself keeps it (server._template_keeps_prior_reasoning).

Naive N0.5's template keeps every earlier turn's reasoning (keep_all_reasoning defaults to true) and changes only the
generation suffix for Off. The server stripped that reasoning on every Off request, which rewrote the history from the
first assistant turn: in a 41k-token chat each thinking toggle re-prefilled everything (cached 0, ~70 s TTFT).
"""
from types import SimpleNamespace

import jinja2
import pytest

from vmlx_engine import server

KEEPS = ("{% for m in messages %}<{{ m.role }}>{% if m.role == 'assistant' and m.reasoning_content %}"
         "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}{% endfor %}"
         "{% if add_generation_prompt %}<assistant>{% if not enable_thinking %}<think></think>{% endif %}{% endif %}")
DROPS = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
         "{% if add_generation_prompt %}<assistant>{% if not enable_thinking %}<think></think>{% endif %}{% endif %}")


class _Tok:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        return jinja2.Template(self.chat_template).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                                          **kwargs)


HISTORY = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "visible", "reasoning_content": "private"},
    {"role": "user", "content": "second"},
]


@pytest.fixture
def engine(monkeypatch):
    def install(template):
        monkeypatch.setattr(server, "_engine", SimpleNamespace(tokenizer=_Tok(template), model=None))
        server._PRIOR_REASONING_PROBE.clear()
    yield install
    server._PRIOR_REASONING_PROBE.clear()


def test_template_that_keeps_prior_reasoning_keeps_history_identical_on_off(engine):
    engine(KEEPS)
    assert server._template_keeps_prior_reasoning(request_thinking=False) is True
    assert server._strip_prior_reasoning_for_thinking_off(HISTORY) == HISTORY


def test_template_that_drops_prior_reasoning_keeps_the_strip(engine):
    engine(DROPS)
    assert server._template_keeps_prior_reasoning(request_thinking=False) is False
    cleaned = server._strip_prior_reasoning_for_thinking_off(HISTORY)
    assert cleaned[1] == {"role": "assistant", "content": "visible"}


def test_no_loaded_engine_keeps_the_strip(monkeypatch):
    monkeypatch.setattr(server, "_engine", None)
    server._PRIOR_REASONING_PROBE.clear()
    assert server._template_keeps_prior_reasoning(request_thinking=False) is False
    assert "reasoning_content" not in server._strip_prior_reasoning_for_thinking_off(HISTORY)[1]


def test_probe_is_cached_per_template(engine):
    engine(KEEPS)
    server._template_keeps_prior_reasoning(request_thinking=False)
    assert len(server._PRIOR_REASONING_PROBE) == 1
    server._template_keeps_prior_reasoning(request_thinking=False)
    assert len(server._PRIOR_REASONING_PROBE) == 1
