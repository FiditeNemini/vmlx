# SPDX-License-Identifier: Apache-2.0
"""The scheduler must not pass kwargs a generator's insert() cannot take.

Caught LIVE by the release soak (Nemotron-Omni, native MTP): commit 2b8fd9880
added ``gen_prompt_lens`` to the scheduler's single unconditional insert call
and to SingleBatchGenerator/DSV4BatchGenerator — but native-MTP text models
deliberately run the stock ``mlx_lm.BatchGenerator``, whose insert() does not
take it. Every request raised TypeError, the failure handler classified it as
a cache problem and re-queued, and the serve wedged in a retry loop until the
client timed out. The fix probes the ACTIVE generator's signature once and
omits the kwarg for generators that own their own prefill.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_the_stock_batch_generator_still_lacks_the_kwarg():
    """The premise of the probe: if mlx_lm ever grows the parameter, the probe
    starts passing it there automatically and this test documents the flip."""
    from mlx_lm.generate import BatchGenerator

    assert "gen_prompt_lens" not in inspect.signature(
        BatchGenerator.insert
    ).parameters


def test_vmlx_generators_accept_the_kwarg():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    for cls in (SingleBatchGenerator, DSV4BatchGenerator):
        assert "gen_prompt_lens" in inspect.signature(cls.insert).parameters, (
            f"{cls.__name__}.insert lost gen_prompt_lens — the cold-prefill "
            "split silently dies on that path"
        )


def test_scheduler_call_site_is_capability_probed():
    src = (ROOT / "vmlx_engine" / "scheduler.py").read_text(encoding="utf-8")
    call = src.index("uids = self.batch_generator.insert(")
    window = src[call - 1500 : call]
    assert "_insert_accepts_gen_prompt_lens()" in window, (
        "the insert call site passes gen_prompt_lens unconditionally again — "
        "native-MTP text models (stock mlx_lm BatchGenerator) wedge every "
        "request in a TypeError retry loop"
    )
    assert "def _insert_accepts_gen_prompt_lens" in src


def test_probe_behavior_on_both_signature_shapes():
    from vmlx_engine.scheduler import Scheduler

    class _Accepts:
        def insert(self, prompts, max_tokens=None, gen_prompt_lens=None):
            pass

    class _Rejects:
        def insert(self, prompts, max_tokens=None):
            pass

    stub = SimpleNamespace(batch_generator=_Accepts())
    assert Scheduler._insert_accepts_gen_prompt_lens(stub) is True
    # Cached per generator instance; a generator swap re-probes.
    stub.batch_generator = _Rejects()
    assert Scheduler._insert_accepts_gen_prompt_lens(stub) is False
    assert Scheduler._insert_accepts_gen_prompt_lens(stub) is False
