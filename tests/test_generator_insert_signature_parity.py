"""Every generator the LLM scheduler can bind must accept the scheduler's call.

A kwarg was added to the scheduler's batch_generator.insert(...) call for one
generator class. DSV4 binds a different class, so every DSV4 request died with
"DSV4BatchGenerator.insert() got an unexpected keyword argument
'gen_prompt_lens'" and was pushed back onto the waiting queue forever -- while
the full engine suite stayed green, because no test drove that generator's
insert. It was found by serving the model, not by testing it.

These tests compare the call site against each bindable generator's signature,
so adding a kwarg to one and not the others fails here instead of at serve time.
"""

import ast
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _scheduler_insert_kwargs():
    """Keyword names the LLM scheduler passes to batch_generator.insert(...)."""
    source = (REPO / "vmlx_engine" / "scheduler.py").read_text()
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "insert"):
            continue
        target = func.value
        if not (isinstance(target, ast.Attribute) and target.attr == "batch_generator"):
            continue
        found.append([kw.arg for kw in node.keywords if kw.arg is not None])
    return found


def _generator_classes():
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    return [SingleBatchGenerator, DSV4BatchGenerator]


def test_scheduler_call_site_is_discoverable():
    calls = _scheduler_insert_kwargs()
    assert calls, "no self.batch_generator.insert(...) call found to check against"


@pytest.mark.parametrize("cls", _generator_classes(), ids=lambda c: c.__name__)
def test_generator_accepts_every_scheduler_kwarg(cls):
    sig = inspect.signature(cls.insert)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    for kwargs in _scheduler_insert_kwargs():
        for name in kwargs:
            assert accepts_var_kw or name in sig.parameters, (
                f"{cls.__name__}.insert() cannot accept '{name}', which "
                f"vmlx_engine/scheduler.py passes. Serving a model on this "
                f"generator raises TypeError on every request."
            )


def test_gen_prompt_lens_specifically_is_accepted_everywhere():
    """Pin the exact kwarg that broke DSV4, so a revert cannot quietly undo it."""
    for cls in _generator_classes():
        sig = inspect.signature(cls.insert)
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        assert accepts_var_kw or "gen_prompt_lens" in sig.parameters, cls.__name__


def test_dsv4_insert_tolerates_the_kwarg_without_binding_a_model():
    """Signature-level check is not enough on its own -- bind the arguments the
    scheduler actually sends and confirm Python accepts them."""
    from vmlx_engine.utils.dsv4_batch_generator import DSV4BatchGenerator

    sig = inspect.signature(DSV4BatchGenerator.insert)
    bound = sig.bind(
        None,
        [[1, 2, 3]],
        max_tokens=[16],
        caches=None,
        gen_prompt_lens=[2],
    )
    assert bound.arguments["gen_prompt_lens"] == [2]
