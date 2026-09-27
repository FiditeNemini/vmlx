"""CPU-only tests of bounded diagnostic timing and disabled-path behavior."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_PATH = Path(__file__).parents[1] / "vmlx_engine/native_mtp_forward_probe.py"
_SPEC = importlib.util.spec_from_file_location("forward_probe_under_test", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


class Device:
    def __init__(self):
        self.calls = []

    def eval(self, *arrays):
        self.calls.append(("eval", arrays))

    def synchronize(self):
        self.calls.append(("sync",))


class ForwardProbeTests(unittest.TestCase):
    def test_ar_and_mtp_flags_are_independent(self):
        for ar, mtp in (("0", "1"), ("1", "0")):
            with patch.dict(probe.os.environ, {
                "VMLX_AR_FORWARD_PROBE": ar,
                "VMLX_NATIVE_MTP_FORWARD_PROBE": mtp,
            }):
                for phase in ("ar_model", "ar_sample", "head", "verify"):
                    request, device = SimpleNamespace(), Device()
                    record = probe.start_native_mtp_forward_probe(request, phase, device)
                    enabled = ar == "1" if phase.startswith("ar_") else mtp == "1"
                    self.assertEqual(record is not None, enabled)
                    if not enabled:
                        self.assertEqual(device.calls, [])
                        self.assertEqual(vars(request), {})

    @patch.dict(probe.os.environ, {"VMLX_AR_FORWARD_PROBE": "1"})
    def test_ar_phases_stop_at_eight_and_use_distinct_label(self):
        request, device = SimpleNamespace(request_id="ar"), Device()
        output = SimpleNamespace(shape=(1,), dtype="uint32")
        for phase in ("ar_model", "ar_sample"):
            for _ in range(8):
                record = probe.start_native_mtp_forward_probe(request, phase, device)
                with patch.object(probe.logger, "info") as log:
                    record.finish(output)
                    self.assertEqual(log.call_args.args[0], "MLLM AR completed forward %s")
                    self.assertEqual(json.loads(log.call_args.args[1])["phase"], phase)
            before = list(device.calls)
            self.assertIsNone(probe.start_native_mtp_forward_probe(request, phase, device))
            self.assertEqual(device.calls, before)

    def test_disabled_has_no_device_calls_or_request_mutation(self):
        for value in ("0", "", "false"):
            with self.subTest(value=value), patch.dict(
                probe.os.environ, {"VMLX_NATIVE_MTP_FORWARD_PROBE": value}
            ):
                request, device = SimpleNamespace(), Device()
                self.assertIsNone(probe.start_native_mtp_forward_probe(
                    request, "head", device, inputs=(object(),)
                ))
                self.assertEqual(device.calls, [])
                self.assertEqual(vars(request), {})

    @patch.dict(probe.os.environ, {"VMLX_NATIVE_MTP_FORWARD_PROBE": "1"})
    def test_input_wait_is_separate_and_outputs_complete_before_publication(self):
        request, device = SimpleNamespace(request_id="r"), Device()
        source = object()
        output = SimpleNamespace(shape=(1, 2, 32), dtype="float16")
        with patch.object(probe.time, "perf_counter", side_effect=(1., 1.25, 1.75)):
            record = probe.start_native_mtp_forward_probe(
                request, "verify", device, inputs=(source, None), verify_rows=2
            )
            self.assertEqual(device.calls, [("eval", (source,)), ("sync",)])
            with patch.object(probe.logger, "info") as log:
                record.finish(output, None)
                record.finish(output)
                log.assert_called_once()
                published = json.loads(log.call_args.args[1])
        self.assertEqual(published["input_ready_ms"], 250.)
        self.assertEqual(published["forward_ready_ms"], 500.)
        self.assertEqual(published["output_shapes"], [[1, 2, 32]])
        self.assertEqual(published["verify_rows"], 2)
        self.assertTrue(published["perturbs_pipeline"])
        self.assertFalse(published["kernel_time"])
        self.assertEqual(device.calls[-2:], [("eval", (output,)), ("sync",)])
        self.assertEqual(vars(request), {"request_id": "r", "_native_mtp_forward_probe_counts": {"verify": 1}})

    @patch.dict(probe.os.environ, {"VMLX_NATIVE_MTP_FORWARD_PROBE": "1"})
    def test_cap_is_per_request_and_phase_and_adds_no_calls_after_limit(self):
        request, device = SimpleNamespace(), Device()
        for phase in ("head", "verify"):
            for _ in range(8):
                self.assertIsNotNone(probe.start_native_mtp_forward_probe(request, phase, device))
            before = list(device.calls)
            self.assertIsNone(probe.start_native_mtp_forward_probe(request, phase, device))
            self.assertEqual(device.calls, before)
        self.assertIsNotNone(probe.start_native_mtp_forward_probe(SimpleNamespace(), "head", device))

    @patch.dict(probe.os.environ, {"VMLX_NATIVE_MTP_FORWARD_PROBE": "1"})
    def test_failure_is_not_published_as_completed(self):
        request, device = SimpleNamespace(), Device()
        record = probe.start_native_mtp_forward_probe(request, "head", device)
        with patch.object(device, "eval", side_effect=RuntimeError("device error")), patch.object(probe.logger, "info") as log:
            with self.assertRaisesRegex(RuntimeError, "device error"):
                record.finish(object())
            log.assert_not_called()
        self.assertFalse(record.finished)

    @patch.dict(probe.os.environ, {"VMLX_NATIVE_MTP_FORWARD_PROBE": "1"})
    def test_unknown_phase_rejected_before_mutation(self):
        request, device = SimpleNamespace(), Device()
        with self.assertRaises(ValueError):
            probe.start_native_mtp_forward_probe(request, "other", device)
        self.assertEqual(vars(request), {})
        self.assertEqual(device.calls, [])


if __name__ == "__main__":
    unittest.main()
