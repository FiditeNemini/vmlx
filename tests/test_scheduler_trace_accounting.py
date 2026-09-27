"""CPU checks of the production scheduler trace method, without model imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def trace_fixture():
    path = Path(__file__).resolve().parents[1] / "vmlx_engine/mllm_scheduler.py"
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLLMScheduler")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_record_scheduler_trace")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    logs = []
    namespace = {"logger": SimpleNamespace(info=lambda fmt, *args: logs.append(fmt % args))}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    scheduler = SimpleNamespace(_scheduler_trace_timings={})
    return scheduler, lambda output: namespace["_record_scheduler_trace"](scheduler, output, executor_s=.01, dispatch_s=.001), logs


def item(request, tokens, finished=False):
    return SimpleNamespace(request_id=request, new_token_ids=tokens, finished=finished)


def step(*outputs):
    return SimpleNamespace(outputs=list(outputs), trace_timings={"total_s": .009, "batch_next_s": .008})


class SchedulerTraceAccountingTests(unittest.TestCase):
    def test_speculative_burst_counts_one_step_and_all_tokens(self):
        scheduler, record, logs = trace_fixture()
        record(step(item("a", [1]), item("a", [2]), item("a", [3], True)))
        self.assertEqual(len(logs), 1)
        for text in ("steps=1 ", "output_items=3 ", "output_tokens=3 ", "executor_ms=10.000", "step_ms=9.000", "batch_next_ms=8.000"):
            self.assertIn(text, logs[0])
        self.assertEqual(scheduler._scheduler_trace_timings, {})

    def test_interleaved_requests_keep_separate_step_and_item_counts(self):
        scheduler, record, logs = trace_fixture()
        record(step(item("a", [1]), item("b", [8]), item("a", [2])))
        self.assertEqual(scheduler._scheduler_trace_timings["a"]["steps"], 1)
        self.assertEqual(scheduler._scheduler_trace_timings["b"]["steps"], 1)
        record(step(item("a", [3, 4], True), item("b", None, True)))
        self.assertEqual(len(logs), 2)
        for line in logs:
            self.assertIn("steps=2 ", line)
            self.assertIn("executor_ms=20.000", line)
        self.assertIn("output_items=3 output_tokens=4", logs[0])
        self.assertIn("output_items=2 output_tokens=1", logs[1])
        self.assertEqual(scheduler._scheduler_trace_timings, {})

    def test_empty_step_leaves_accumulators_unchanged(self):
        scheduler, record, logs = trace_fixture()
        record(step())
        self.assertEqual(scheduler._scheduler_trace_timings, {})
        self.assertEqual(logs, [])


if __name__ == "__main__":
    unittest.main()
