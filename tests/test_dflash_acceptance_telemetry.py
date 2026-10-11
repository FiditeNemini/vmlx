"""Exercise the production accounting block without importing a model runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_ar_bursts_do_not_count_as_accepted_drafts():
    source = Path("vmlx_engine/models/mllm.py").read_text()
    tree = ast.parse(source)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name)
                and n.iter.func.id == "stream_dflash2_generate")
    # Run only production bookkeeping before terminal reporting/yielding.
    statements = []
    for node in loop.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "last_chunk" for t in node.targets):
            break
        statements.append(node)
    code = compile(ast.Module(body=statements, type_ignores=[]), "accounting", "exec")
    state = dict(token_count=0, emitted_per_cycle=[], drafted_total=0, width_cycles={})
    for emitted, drafted in [(1, 0), (8, 0), (4, 4), (2, 4)]:
        state["chunk"] = SimpleNamespace(tokens=list(range(emitted)), accepted=emitted, drafted=drafted)
        exec(code, state)
    assert state["token_count"] == 15
    assert state["emitted_per_cycle"] == [4, 2]
    assert state["drafted_total"] == 8
    assert state["width_cycles"] == {5: 2}
    assert sum(n - 1 for n in state["emitted_per_cycle"]) / state["drafted_total"] == 0.5
