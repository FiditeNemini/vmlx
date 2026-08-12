# SPDX-License-Identifier: Apache-2.0
"""Both schedulers must derive the L2 namespace from the SAME recipe.

Block hashes are tokens+parent only, so model identity lives ENTIRELY in the
namespace. The text scheduler grew a `bundle=` weight/config fingerprint
precisely so an in-place bundle replacement could not refault stale tensors;
the MLLM scheduler kept a second, weaker copy that omitted it — along with the
zaya scope and the looped-model identity — and built its BlockDiskStore without
`expected_num_layers`, which is what disarms the wrong-model record validator.

Re-quantizing a VLM bundle in place (same path, new weights — the canonical
swap workflow) therefore kept the same namespace and replayed KV computed by
the old weights. The per-record fingerprint does not save it: that covers
RUNTIME drift, not MODEL drift.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "vmlx_engine" / "scheduler.py"
MLLM_SCHEDULER = ROOT / "vmlx_engine" / "mllm_scheduler.py"
PREFIX_CACHE = ROOT / "vmlx_engine" / "prefix_cache.py"


def test_both_schedulers_call_the_shared_builder():
    for path in (SCHEDULER, MLLM_SCHEDULER):
        assert "build_block_cache_namespace(" in path.read_text(), path.name


def test_neither_scheduler_hand_rolls_a_namespace():
    """A locally-assembled scope key is how the two copies drifted apart."""
    offenders = []
    for path in (SCHEDULER, MLLM_SCHEDULER):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(r"block_scope_key\s*=\s*\(", line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"hand-rolled block_scope_key: {offenders}"


def test_the_builder_carries_every_identity_component():
    """Each of these was present on one path and missing on the other."""
    source = PREFIX_CACHE.read_text()
    body = source[source.index("def build_block_cache_namespace(") :]
    body = body[: body.index("\ndef ", 10)]
    for component in (
        "bundle=",                      # weight/config fingerprint
        "paged_cache_schema=",
        "runtime_cache_fingerprint()",
        "looped_cache_identity_scope",  # looped-model identity
        "dsv4_scope",
        "zaya_scope",
        "quant=",
        "tq_native=",
    ):
        assert component in body, f"builder dropped {component!r}"


def test_the_bundle_fingerprint_actually_reaches_the_key():
    """The whole point: identity must be content-derived, not path-derived."""
    source = PREFIX_CACHE.read_text()
    body = source[source.index("def build_block_cache_namespace(") :]
    body = body[: body.index("\ndef ", 10)]
    assert "compute_model_cache_key(" in body
    assert "f\":bundle={bundle_cache_key}\"" in body


def test_mllm_store_arms_the_wrong_model_validator():
    """`expected_num_layers=None` means 'skip the check' in BlockDiskStore."""
    source = MLLM_SCHEDULER.read_text()
    construction = source[source.index("block_disk_store = BlockDiskStore(") :]
    construction = construction[: construction.index(")\n")]
    assert "expected_num_layers=" in construction, (
        "the MLLM BlockDiskStore does not pass expected_num_layers, so the "
        "wrong-model record validator is disarmed on that path"
    )


def test_layer_count_helper_is_shared_not_duplicated():
    assert "def expected_cache_layer_count(" in PREFIX_CACHE.read_text()
    # the text scheduler's method must delegate, not re-implement
    source = SCHEDULER.read_text()
    method = source[source.index("def _expected_cache_layer_count(") :]
    method = method[: method.index("\n    def ", 10)]
    assert "expected_cache_layer_count(" in method
    # The tell for a re-implementation is the probing itself, not the words in
    # the docstring — which legitimately explains why num_hidden_layers is the
    # wrong number to compare against.
    code = "\n".join(
        line for line in method.splitlines() if not line.strip().startswith('"')
    )
    assert "make_cache()" not in code, "re-implemented instead of delegating"
