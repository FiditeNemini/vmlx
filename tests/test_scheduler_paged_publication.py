# SPDX-License-Identifier: Apache-2.0
"""Execute the text cleanup's actual store block with controlled L2 receipts."""
import ast
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vmlx_engine.persistence_outcome import TerminalPersistenceLedger


@pytest.mark.parametrize("retained", [None, 0, 32, 63])
def test_text_cleanup_uses_publication_boundary_before_companion(retained):
    tree = ast.parse((Path(__file__).parents[1] / "vmlx_engine/scheduler.py").read_text())
    # Execute, rather than duplicate, the production store + companion + ledger
    # statements. Tensor extraction and inference are outside this unit's scope.
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If) and any(
        isinstance(stmt, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_stored_block_table"
            for target in stmt.targets
        ) for stmt in node.body
    )]
    assert len(blocks) == 1
    code = compile(ast.Module(body=blocks, type_ignores=[]), "scheduler.py", "exec")
    table = None if retained is None else SimpleNamespace(
        num_tokens=retained, block_ids=[1] if retained else []
    )
    owner = SimpleNamespace(
        _is_hybrid=False,
        block_aware_cache=SimpleNamespace(store_cache=MagicMock(return_value=table)),
        _pick_cache_type_for_request=lambda request: "assistant",
        _retarget_ssm_rederive_to_paged_boundary=MagicMock(),
        _persist_hybrid_ssd_companion=MagicMock(),
        _dsv4_trace_timing=MagicMock(),
    )
    ledger = TerminalPersistenceLedger()
    exec(code, {
        "self": owner, "request_id": "store-test",
        "request": SimpleNamespace(_extracted_cache=[object()]),
        "cache_data": [object()], "store_tokens": list(range(63)),
        "prompt_tokens": list(range(64)), "cache_key_override": list(range(63)),
        "_PERSIST": ledger, "time": time, "logger": logging.getLogger(__name__),
    })
    outcome = ledger.take("store-test")
    assert outcome["retained_tokens"] == retained
    if retained:
        assert outcome["outcome"] == "stored"
        owner._persist_hybrid_ssd_companion.assert_called_once()
        assert owner._persist_hybrid_ssd_companion.call_args.args[-1] is table
        assert str(retained) in outcome["detail"]
    else:
        assert outcome["outcome"] == "refused"
        owner._persist_hybrid_ssd_companion.assert_not_called()
        owner._retarget_ssm_rederive_to_paged_boundary.assert_not_called()
