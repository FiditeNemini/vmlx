# SPDX-License-Identifier: Apache-2.0
"""The vendored Muse runtime must win over a partial upstream package.

mlx-vlm 0.6.x ships its own ``mlx_vlm.models.muse_glimmer``, but names its
preprocessing module ``processing_muse_glimmer`` rather than the ``processor``
this loader imports. The original "if the name imports, upstream owns it" rule
therefore broke every fresh pip install: the server died with
ModuleNotFoundError on ``mlx_vlm.models.muse_glimmer.processor``, and had it
loaded it would have used a forward pass without this port's corrections.
The bundled app hid the bug by pinning mlx-vlm 0.5.0, which has no upstream
package at all.
"""

import importlib.util

from vmlx_engine.models import muse_glimmer_register as reg


def test_upstream_without_processor_is_rejected(monkeypatch):
    """A partial upstream package must NOT be accepted."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name == reg._PACKAGE_NAME:
            return object()          # upstream package "exists"
        if name == f"{reg._PACKAGE_NAME}.processor":
            return None              # ...but has no processor
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert reg._upstream_is_usable() is False


def test_upstream_with_processor_is_accepted(monkeypatch):
    """A genuine drop-in upstream package is still honoured."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name in (reg._PACKAGE_NAME, f"{reg._PACKAGE_NAME}.processor"):
            return object()
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert reg._upstream_is_usable() is True


def test_registration_exposes_the_processor_the_loader_imports():
    """End state: the loader's exact import must resolve to the vendored copy."""
    reg.register_muse_glimmer_runtime()
    from mlx_vlm.models.muse_glimmer.processor import (  # noqa: F401
        MuseGlimmerImageProcessor,
        MuseGlimmerProcessor,
        MuseGlimmerVideoProcessor,
    )

    import mlx_vlm.models.muse_glimmer as mg

    assert "vmlx_engine" in mg.__file__, "vendored runtime must own the namespace"
