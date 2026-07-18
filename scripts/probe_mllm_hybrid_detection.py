#!/usr/bin/env python3
"""Load an MLLM through the engine path and report hybrid-cache ownership.

This is intentionally a load-only, offline diagnostic.  It uses
``MLXMultimodalLM.load()`` -- the same wrapper and JANG-affine VLM loader used
by ``BatchedEngine._start_mllm`` -- and never starts or contacts a server.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# The managed diagnostic sandbox can expose the CPU MLX backend while denying
# the Metal device.  Select CPU and turn compile decorators into pass-throughs
# before importing mlx.nn-backed engine modules.  This changes execution device
# only; loader selection, wrappers, and make_cache ownership stay identical.
if "--cpu" in sys.argv:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)

    def _passthrough_compile(function=None, **_kwargs):
        if function is None:
            return lambda inner: inner
        return function

    mx.compile = _passthrough_compile

from vmlx_engine.mllm_batch_generator import (  # noqa: E402, I001
    _hybrid_cache_layout,
    _is_kv_like,
)
from vmlx_engine.models.mllm import MLXMultimodalLM  # noqa: E402


_INNER_MODEL_ATTRS = (
    "language_model",
    "model",
    "inner",
    "base_model",
    "text_model",
    "transformer",
)


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _walk_model_levels(root: Any, *, max_depth: int = 5) -> Iterator[tuple[str, Any]]:
    """Walk only model-wrapper attributes, avoiding layers and parameter trees."""

    queue: deque[tuple[str, Any, int]] = deque([("model", root, 0)])
    seen: set[int] = set()
    while queue:
        label, value, depth = queue.popleft()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        yield label, value
        if depth >= max_depth:
            continue
        for attr in _INNER_MODEL_ATTRS:
            try:
                child = getattr(value, attr, None)
            except Exception:
                continue
            if child is not None and child is not value:
                queue.append((f"{label}.{attr}", child, depth + 1))


def _cache_template(value: Any) -> tuple[list[Any] | None, str | None]:
    make_cache = getattr(value, "make_cache", None)
    if not callable(make_cache):
        return None, None
    try:
        return list(make_cache() or []), None
    except Exception as exc:  # pragma: no cover - diagnostic only
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet-loader", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet_loader else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    wrapper = MLXMultimodalLM(str(args.model))
    wrapper.load()
    model = wrapper.model
    language_model = getattr(model, "language_model", model)

    print(f"bundle={args.model.resolve()}")
    print(f"type(model)={_type_name(model)}")
    print(f"type(language_model)={_type_name(language_model)}")
    print(f"model_has_make_cache={hasattr(model, 'make_cache')}")
    print(f"language_model_has_make_cache={hasattr(language_model, 'make_cache')}")

    direct_template, direct_error = _cache_template(language_model)
    direct_positions = None
    direct_num_layers = None
    if direct_template is not None:
        direct_num_layers = len(direct_template)
        direct_positions = [
            index for index, cache in enumerate(direct_template) if _is_kv_like(cache)
        ]
    direct_is_hybrid = bool(
        direct_positions is not None
        and direct_num_layers is not None
        and len(direct_positions) < direct_num_layers
    )
    print(f"legacy_detection_error={direct_error!r}")
    print(f"legacy_hybrid_kv_positions={direct_positions!r}")
    print(f"legacy_hybrid_num_layers={direct_num_layers!r}")
    print(f"legacy_is_hybrid={direct_is_hybrid}")

    owner, owner_path, template, positions, error = _hybrid_cache_layout(
        model, language_model
    )
    resolved_is_hybrid = bool(
        positions is not None
        and template is not None
        and len(positions) < len(template)
    )
    print(f"resolved_make_cache_owner={owner_path!r}")
    owner_type = _type_name(owner) if owner is not None else None
    print(f"resolved_make_cache_owner_type={owner_type!r}")
    print(f"resolved_detection_error={error!r}")
    print(f"resolved_hybrid_kv_positions={positions!r}")
    num_layers = len(template) if template is not None else None
    print(f"resolved_hybrid_num_layers={num_layers!r}")
    print(f"resolved_is_hybrid={resolved_is_hybrid}")

    print("levels:")
    for label, value in _walk_model_levels(model):
        make_cache = getattr(value, "make_cache", None)
        print(
            f"  {label}: type={_type_name(value)} "
            f"has_make_cache={hasattr(value, 'make_cache')} "
            f"callable_make_cache={callable(make_cache)}"
        )
        template, error = _cache_template(value)
        if error is not None:
            print(f"    make_cache_error={error}")
            continue
        if template is None:
            continue
        names = [type(cache).__name__ for cache in template]
        kv_like = [_is_kv_like(cache) for cache in template]
        positions = [index for index, flag in enumerate(kv_like) if flag]
        print(f"    template_class_names={names!r}")
        print(f"    template_is_kv_like={kv_like!r}")
        print(f"    hybrid_kv_positions={positions!r}")
        print(f"    hybrid_num_layers={len(template)}")
        print(f"    is_hybrid={len(positions) < len(template)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
