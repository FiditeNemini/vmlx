"""A monkey-patch that silently misses its target is worse than no patch.

Every mlx_vlm compat patch is defensive: it try/except-imports its target and
returns quietly if the module, class or method is not where it expects. That
is right for robustness and wrong for testability — when upstream renames
something, the patch becomes a no-op and the whole LFM2/Qwen VL compat test
file goes GREEN by skipping, because every test in it uses importorskip.

Users find out instead of us. LFM2-VL image prompts start dying on a
signature mismatch; bundles with llama-style MLP names stop loading.

This file fails CLOSED. If mlx_vlm is absent entirely it skips (a legitimate
dev environment). If mlx_vlm is present but a patch did not take, that means
upstream moved and our patch is dead — and it FAILS.
"""

import pytest

# (module path, attribute chain to the patched callable, sentinel attribute)
PATCH_TARGETS = [
    (
        "mlx_vlm.models.lfm2_vl.lfm2_vl",
        ("Model", "get_input_embeddings"),
        "_vmlx_lfm2_vl_positional",
    ),
    (
        "mlx_vlm.models.lfm2_vl.lfm2_vl",
        ("Model", "sanitize"),
        "_vmlx_lfm2_mlp_aliases",
    ),
]


def _resolve(module, chain):
    obj = module
    for name in chain:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


@pytest.mark.parametrize("module_path,chain,sentinel", PATCH_TARGETS)
def test_patch_target_still_exists_and_patch_took(module_path, chain, sentinel):
    pytest.importorskip("mlx_vlm", reason="mlx_vlm not installed in this env")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError:  # pragma: no cover - upstream moved the module
        pytest.fail(
            f"mlx_vlm is installed but {module_path} is gone. The vMLX compat "
            f"patch that targets it is now a silent no-op — upstream moved, "
            f"and the LFM2/Qwen VL tests would go green by skipping."
        )

    target = _resolve(module, chain)
    assert target is not None, (
        f"{module_path}.{'.'.join(chain)} no longer exists; the compat patch "
        f"targeting it silently did nothing."
    )
    assert getattr(target, sentinel, False), (
        f"{module_path}.{'.'.join(chain)} exists but is missing {sentinel!r}: "
        f"mlx_vlm_compat.apply() ran and the patch did NOT take. Upstream has "
        f"probably changed this method, so the behaviour we depend on is gone."
    )
