# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned Muse Glimmer runtime.

Muse Glimmer bundles use ``model_type=muse_glimmer`` with a vision+video tower.
Neither mlx-vlm nor mlx-lm ships this architecture, and the bundles carry no
remote code, so vMLX vendors the runtime and installs it under the upstream
``mlx_vlm.models.muse_glimmer`` namespace at VLM-load time.

Idempotent. It defers to upstream ONLY when upstream actually provides what the
loader needs.

mlx-vlm 0.6.x began shipping its own ``muse_glimmer`` package, and the earlier
"if the name imports, upstream owns it" rule handed the namespace straight to it.
That broke every fresh pip install two ways: upstream names its preprocessing
module ``processing_muse_glimmer`` rather than ``processor``, so the loader's
``from mlx_vlm.models.muse_glimmer.processor import ...`` raised
ModuleNotFoundError and the server refused to start; and upstream's forward pass
does not carry the four divergences this port fixes (centered RMSNorm, weightless
QK-norm with the query-side scale, the NoPE arrangement, and per-type SWA masks),
so even a load that succeeded would emit fluent nonsense. The bundled app was
unaffected because it pins mlx-vlm 0.5.0, which has no upstream package — which
is exactly why the app tested clean while the wheel did not.

So the vendored runtime now wins by default, and upstream is accepted only if it
exposes the ``processor`` module this loader imports.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE_NAME = "mlx_vlm.models.muse_glimmer"
_VENDORED_DIR = Path(__file__).resolve().parent / "muse_glimmer"


def muse_glimmer_runtime_available() -> bool:
    """Return True when an upstream or vendored Muse Glimmer runtime exists."""
    if _PACKAGE_NAME in sys.modules:
        return True
    if importlib.util.find_spec(_PACKAGE_NAME) is not None:
        return True
    return (_VENDORED_DIR / "__init__.py").is_file()


def _upstream_is_usable() -> bool:
    """True when an upstream mlx-vlm package supplies the loader's `processor`.

    Checked by spec lookup rather than import so a partial upstream package is
    never executed and left half-registered in ``sys.modules``.
    """
    try:
        if importlib.util.find_spec(_PACKAGE_NAME) is None:
            return False
        return importlib.util.find_spec(f"{_PACKAGE_NAME}.processor") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def register_muse_glimmer_runtime() -> bool:
    """Install the vendored Muse Glimmer modules into ``sys.modules`` if needed."""
    global _REGISTERED
    if _REGISTERED:
        return True

    init_path = _VENDORED_DIR / "__init__.py"
    have_vendored = init_path.is_file()

    # Only hand the namespace to upstream when it can actually serve this
    # loader: it must expose the `processor` submodule we import by name. An
    # upstream package that lacks it (mlx-vlm 0.6.x names it
    # `processing_muse_glimmer`) is NOT a drop-in, and taking it would also lose
    # this port's forward-pass corrections.
    if not have_vendored or _upstream_is_usable():
        try:
            importlib.import_module(_PACKAGE_NAME)
            _REGISTERED = True
            logger.debug("Muse Glimmer runtime already provided by mlx-vlm")
            return False
        except ModuleNotFoundError:
            pass

    if not have_vendored:
        return False

    # Displace a partial upstream package so the vendored runtime is what the
    # rest of the process resolves.
    for name in [n for n in sys.modules if n == _PACKAGE_NAME or n.startswith(_PACKAGE_NAME + ".")]:
        sys.modules.pop(name, None)

    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(_VENDORED_DIR)],
    )
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    try:
        parent = importlib.import_module("mlx_vlm.models")
        setattr(parent, "muse_glimmer", module)
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PACKAGE_NAME, None)
        raise

    _REGISTERED = True
    logger.info(
        "Registered source-owned Muse Glimmer runtime under "
        "mlx_vlm.models.muse_glimmer"
    )
    return True


_register_muse_glimmer_runtime = register_muse_glimmer_runtime
