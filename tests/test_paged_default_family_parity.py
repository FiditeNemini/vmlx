"""Pins CLI paged-default family policy to the Electron registry.

The generic paged-ON default excludes families whose native cache lane breaks
under paged, and skips MLLM loads. gemma4's typed mixed-SWA cache was
live-proven paged (2026-08-08) and the app keeps it paged-ON even for
multimodal loads, so the CLI must not exclude it — otherwise a bare
`vmlx-serve serve` diverges from the UI.
"""

from pathlib import Path

from vmlx_engine import cli


class TestPagedDefaultFamilyParity:
    def test_incompatible_set_excludes_only_m3_and_openpangu(self):
        assert cli._PAGED_INCOMPATIBLE_FAMILIES == {
            "minimax_m3",
            "minimax_m3_vl",
            "openpangu_v2",
        }

    def test_mllm_paged_exempt_set_matches_the_panel(self):
        """Bare-CLI paged default must match what the panel opts into.

        This set exists so an MLLM family the app runs PAGED is not silently
        run UNPAGED from the command line. Muse Glimmer rides the same typed
        mixed-SWA paged lane as gemma4 and the panel registers it
        usePagedCache: true, so it belongs here; leaving it out reproduced
        exactly the divergence the set was created to prevent.
        """
        assert cli._PAGED_MLLM_EXEMPT_FAMILIES == {
            "gemma4",
            "gemma4_text",
            "muse_glimmer",
        }

    def test_panel_opts_muse_into_paged_too(self):
        registry = (
            Path(__file__).resolve().parents[1]
            / "panel"
            / "src"
            / "main"
            / "model-config-registry.ts"
        )
        src = registry.read_text()
        idx = src.index("registerFamily('muse-glimmer'")
        assert "usePagedCache: true" in src[idx : idx + 600]

    def test_panel_registry_still_opts_gemma4_into_paged(self):
        registry = (
            Path(__file__).resolve().parents[1]
            / "panel"
            / "src"
            / "main"
            / "model-config-registry.ts"
        )
        src = registry.read_text()
        idx = src.index("next.family === 'gemma4'")
        window = src[idx : idx + 600]
        assert "next.usePagedCache = true" in window
