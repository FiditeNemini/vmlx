# SPDX-License-Identifier: Apache-2.0
"""vmlx#262: doctor's inference test must apply the native-MTP sanitize
patches before building the model tree, exactly like the serve path.

Without them, a native-MTP bundle (stacked mtp.layers.N.mlp.switch_mlp.*)
gets a per-expert module tree and weight binding KeyErrors on
'language_model.mtp.layers.0.mlp.experts.0.gate_proj.weight' -- a
doctor-only false negative while serve loads the identical checkpoint
fine (same defect class as #213, which fixed serve but not doctor).
"""

from unittest.mock import patch


class TestDoctorNativeMtpPreload:
    def test_inference_test_applies_native_mtp_before_load(self):
        from vmlx_engine.commands.doctor import _run_inference_test

        call_order = []

        def fake_mtp(path, *, allow_runtime=True, reason=None):
            call_order.append(("mtp", str(path), allow_runtime))
            return {}

        def fake_load(path, tokenizer_config=None, skip_turboquant=False):
            call_order.append(("load", str(path)))
            raise RuntimeError("stop before real load")

        with patch("vmlx_engine.native_mtp.maybe_apply_native_mtp", fake_mtp), \
             patch("vmlx_engine.utils.tokenizer.load_model_with_fallback", fake_load), \
             patch("vmlx_engine.utils.nemotron_latent_moe.ensure_latent_moe_support",
                   lambda p: None):
            success, message = _run_inference_test("/fake/native-mtp-bundle")

        assert not success and "stop before real load" in message
        assert call_order[0] == ("mtp", "/fake/native-mtp-bundle", False), (
            "maybe_apply_native_mtp must run BEFORE the model load, with "
            "allow_runtime=False (doctor is a smoke test: sanitize patches "
            "yes, MTP decode runtime no -- same philosophy as its "
            "skip_turboquant=True)"
        )
        assert call_order[1][0] == "load"

    def test_mtp_autodetect_failure_never_breaks_doctor(self):
        from vmlx_engine.commands.doctor import _run_inference_test

        def broken_mtp(path, *, allow_runtime=True, reason=None):
            raise RuntimeError("inspection exploded")

        def fake_load(path, tokenizer_config=None, skip_turboquant=False):
            raise RuntimeError("reached the load")

        with patch("vmlx_engine.native_mtp.maybe_apply_native_mtp", broken_mtp), \
             patch("vmlx_engine.utils.tokenizer.load_model_with_fallback", fake_load), \
             patch("vmlx_engine.utils.nemotron_latent_moe.ensure_latent_moe_support",
                   lambda p: None):
            success, message = _run_inference_test("/fake/bundle")

        assert not success and "reached the load" in message, (
            "a failing MTP autodetect must be swallowed (debug log) and the "
            "inference test must still proceed to the load"
        )
