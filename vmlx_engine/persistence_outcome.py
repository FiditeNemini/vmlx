# SPDX-License-Identifier: Apache-2.0
"""Per-request terminal persistence outcome (both scheduler lanes).

The terminal durability barrier used to log ``cache_persisted=true`` as soon
as the cleanup fence was released, but the fence is released in a ``finally``
whether the store succeeded, was skipped, was refused for budget, or raised.
Cleanup completion is not evidence of durable storage.  Every store site now
records what actually happened for the request and the barrier reports THAT;
a request nothing recorded for is reported as ``unknown``.

Outcomes:
  stored           a prefix/cache record was written for this request
  already_durable  nothing written because the cached prefix already covered
                   the request's reusable boundary
  skipped          nothing to store (no extracted cache, empty resolved cache,
                   output too short, no prompt token ids, ...)
  refused          a store was attempted and rejected (budget, unsafe record)
  failed           the store raised
  unknown          cleanup completed but no site recorded an outcome
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

OUTCOMES = ("stored", "already_durable", "skipped", "refused", "failed", "unknown")
_MAX_ENTRIES = 4096


class TerminalPersistenceLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def record(
        self,
        request_id: Any,
        outcome: str,
        detail: str = "",
        *,
        retained_tokens: Optional[int] = None,
        durable: Optional[bool] = None,
    ) -> None:
        """Record the outcome for ``request_id``.  A later ``stored`` never
        downgrades to ``skipped`` (a request can skip one tier and store in
        another); ``failed`` and ``refused`` always win over ``skipped``."""
        if outcome not in OUTCOMES or request_id is None:
            return
        rid = str(request_id)
        with self._lock:
            prev = self._entries.get(rid)
            if prev is not None:
                rank = {"unknown": 0, "skipped": 1, "already_durable": 2, "stored": 3, "refused": 4, "failed": 5}
                if rank[prev["outcome"]] > rank[outcome] and prev["outcome"] in ("stored", "refused", "failed"):
                    return
            self._entries[rid] = {
                "outcome": outcome,
                "detail": str(detail or "")[:200],
                "retained_tokens": retained_tokens,
                "durable": durable,
            }
            self._entries.move_to_end(rid)
            while len(self._entries) > _MAX_ENTRIES:
                self._entries.popitem(last=False)

    def take(self, request_id: Any) -> Dict[str, Any]:
        with self._lock:
            entry = self._entries.pop(str(request_id), None)
        return entry or {"outcome": "unknown", "detail": "cleanup completed; no store outcome recorded", "retained_tokens": None, "durable": None}

    def peek(self, request_id: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._entries[str(request_id)]) if str(request_id) in self._entries else None


LEDGER = TerminalPersistenceLedger()


def format_outcome(entry: Dict[str, Any]) -> str:
    parts = [f"cache_outcome={entry.get('outcome', 'unknown')}"]
    if entry.get("retained_tokens") is not None:
        parts.append(f"retained_tokens={entry['retained_tokens']}")
    if entry.get("durable") is not None:
        parts.append(f"durable={'true' if entry['durable'] else 'false'}")
    if entry.get("detail"):
        parts.append(f"detail={entry['detail']!r}")
    return " ".join(parts)
