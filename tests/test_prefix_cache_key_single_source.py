"""The prefix cache key must be derived in ONE place.

The key strips the generation-prompt suffix, because chat templates append
assistant-role tokens that differ every turn and would otherwise cause a 100%
miss rate. That rule lived as three hand-copied copies.

Keeping one copy is not tidiness: the boundary this returns is also where a warm
turn resumes computing, so anything reproducing a warm turn's arithmetic has to
agree with it exactly. Deriving that width from `_gen_prompt_len` alone
regressed Laguna-S, whose cache covers the whole prompt and whose correct width
is 0.
"""

import inspect

import pytest

from vmlx_engine.scheduler import prefix_cache_key_tokens


class _Req:
    def __init__(self, n_tokens: int, gen_prompt_len: int):
        self.prompt_token_ids = list(range(n_tokens))
        self._gen_prompt_len = gen_prompt_len


@pytest.mark.parametrize(
    "n,gpl,expected",
    [
        (10, 3, 7),      # strips the suffix
        (10, 0, 10),     # no suffix -> untouched
        (10, 10, 10),    # would strip everything -> guarded, untouched
        (10, 99, 10),    # longer than the prompt -> guarded
        (0, 5, 0),       # empty prompt
        (1, 1, 1),       # single token -> guarded
    ],
)
def test_key_length(n, gpl, expected):
    assert len(prefix_cache_key_tokens(_Req(n, gpl))) == expected


def test_missing_attributes_do_not_raise():
    class Bare:
        pass

    assert prefix_cache_key_tokens(Bare()) == []


def test_returns_a_copy_not_the_request_list():
    req = _Req(5, 0)
    out = prefix_cache_key_tokens(req)
    out.append(999)
    assert req.prompt_token_ids == list(range(5)), (
        "the key derivation must not alias the request's token list"
    )


def test_no_site_re_inlines_the_stripping():
    """Fail if a caller hand-rolls the strip again instead of calling this."""
    from vmlx_engine import scheduler

    src = inspect.getsource(scheduler)
    # The pure derivation is `tokens[:-gen_prompt_len]` guarded by a range check.
    # ONE occurrence is legitimate and deliberately not converted: the
    # mixed-attention fetch path (~scheduler.py:3788) is a DIFFERENT rule -- it
    # is additionally gated on _mixed_attention_cache_model and returns
    # fetch_tokens + suffix rather than the key, so folding it in would change
    # behaviour. prefix_cache_key_tokens itself spells the slice with `gpl`, so
    # it does not count here.
    inlined = src.count("[:-gen_prompt_len]") + src.count("[:-_gpl_s]")
    assert inlined <= 1, (
        f"found {inlined} inlined generation-prompt strips in scheduler.py; "
        "route them through prefix_cache_key_tokens() instead"
    )
