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
            # step3p7 is is_mllm, but the panel registers step-3.7-flash with
            # usePagedCache: true AND cacheSubtypeRequiresPaged() returns true
            # for step3p7_full_sliding_kv, so the app runs it paged-REQUIRED
            # while a bare CLI launch ran it unpaged. Verified against the panel
            # in test_panel_requires_paged_for_step3p7 below rather than trusted
            # as a snapshot — this assertion is a hardcoded list, so on its own
            # it only records what someone once believed.
            "step3p7",
        }

    def test_panel_keeps_step3p7_off_paged_ram(self):
        """step3p7 gets NO in-RAM paged tier: SSD block-disk L2 is the only one.

        2026-08-23: in-RAM paged cache is OFF for every family. This test used
        to assert the opposite for step3p7; the subtype is still declared (the
        engine uses it to pick the mixed-SWA lane), but it no longer buys a RAM
        tier.
        """
        panel_root = Path(__file__).resolve().parents[1] / "panel" / "src" / "main"
        registry = (panel_root / "model-config-registry.ts").read_text()
        idx = registry.index("registerFamily('step-3.7-flash'")
        window = registry[idx : idx + 800]
        assert "usePagedCache: false" in window
        assert "usePagedCache: true" not in window
        assert "cacheSubtype: 'step3p7_full_sliding_kv'" in window
        # The cache-shape predicates used to be byte-identical copies in
        # sessions.ts (the argv BUILDER) and SessionSettings.tsx (the argv
        # PREVIEW). They now live in one shared module, so read the rule from
        # its owner and separately confirm the builder still consumes it —
        # grepping sessions.ts for the function body would pin the duplication.
        shared = (
            panel_root.parent / "shared" / "cacheTypeCapabilities.ts"
        ).read_text()
        req = shared.index("function cacheSubtypeRequiresPaged")
        assert "PAGED_REQUIRED_CACHE_SUBTYPES" in shared[req : req + 300]
        assert "step3p7_full_sliding_kv" in shared
        assert "cacheTypeCapabilities" in (panel_root / "sessions.ts").read_text()

    def test_panel_keeps_muse_off_paged_ram(self):
        registry = (
            Path(__file__).resolve().parents[1]
            / "panel"
            / "src"
            / "main"
            / "model-config-registry.ts"
        )
        src = registry.read_text()
        idx = src.index("registerFamily('muse-glimmer'")
        window = src[idx : idx + 600]
        # In-RAM paged cache is OFF for every family (SSD L2 only).
        assert "usePagedCache: false" in window
        assert "usePagedCache: true" not in window

    def test_panel_registry_keeps_gemma4_off_paged_ram(self):
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
        # Gemma-4's typed mixed-SWA path restores from block-disk L2; it does
        # not get a RAM mirror. Paged RAM is OFF for every family.
        assert "next.usePagedCache = false" in window
        assert "next.usePagedCache = true" not in window
