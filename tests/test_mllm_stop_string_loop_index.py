"""The MLLM response loop index must only be moved by loop control (audit 2026-10-07).

`MLLMScheduler._process_batch_responses` walks `responses` with `idx`. Since v1.5.39 the string-stop check assigned
`idx = full_text.find(stop, ...)`: a miss (-1) reset the loop to the same response, which re-added one token to the
detokenizer forever -- every request carrying a `stop` list hung the server (Flash-Next 4S: 126k chars of "```" after
25 s, /health dead). A hit set `idx` to a CHARACTER offset and skipped other rows' responses.
Also: streamed deltas leaked the first characters of a stop string before it matched.
"""
import ast
import inspect
import textwrap

from vmlx_engine.mllm_scheduler import MLLMScheduler


def _loop_index_writes():
    src = textwrap.dedent(inspect.getsource(MLLMScheduler._process_batch_responses))
    fn = ast.parse(src).body[0]
    writes = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and name.id == "idx":
                        writes.append(ast.unparse(node.value))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "idx":
            writes.append(f"+= {ast.unparse(node.value)}")
        elif isinstance(node, (ast.For, ast.comprehension)):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name) and name.id == "idx":
                    writes.append("for-target")
    return writes


def test_only_loop_control_moves_the_response_index():
    writes = _loop_index_writes()
    assert set(writes) <= {"0", "coalesced_end", "+= 1"}, writes
    assert "0" in writes and "+= 1" in writes


def test_streamed_stop_holds_back_partial_stop_prefixes():
    """A streamed delta cannot be retracted: text that may start a stop string must be held until it resolves.

    Live (Flash-Next 4S, before): stop "\\n17" streamed "...16\\n1"; stop "23\\n24" streamed "...23\\n2".
    """
    src = inspect.getsource(MLLMScheduler._process_batch_responses)
    assert "request._stop_emitted = emitted" in src
    assert "full_text.endswith(s[:size])" in src          # the hold-back
    assert "final_text_delta = detok.text[_emitted:]" in src  # held tail released at a non-stop finish
    assert "idx = full_text.find" not in src
