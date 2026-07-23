# SPDX-License-Identifier: Apache-2.0
"""Stable discriminators shared by every prompt-cache backend."""

from __future__ import annotations

import json
from typing import Any, Optional


def canonical_cache_extra_marker(cache_extra_keys: Optional[Any]) -> Optional[str]:
    """Return a stable, comparable marker for non-token cache key inputs."""
    if cache_extra_keys is None:
        return None
    try:
        return json.dumps(
            cache_extra_keys,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return repr(cache_extra_keys)
