"""vmlx#246: every JANG-loader nn.quantize predicate must floor on to_quantized.

Issue #246 proposed adding a hasattr(m, "to_quantized") guard to the VLM-path
get_class_predicate. The guard has been present at ALL predicate sites since
v1.5.1 — including at the reporter's pinned commit 693b2d08, where it is the
first statement of the VLM fast-path predicate (the issue's quoted snippet
omitted it). This test locks the floor in for every current and future
class_predicate closure in jang_loader so a refactor cannot silently drop it:
a predicate that returns truthy for a module without to_quantized makes
nn.quantize abort the entire model load (mlx>=0.31: ValueError "Unable to
quantize model of type ...").
"""

import ast
import inspect

import vmlx_engine.utils.jang_loader as jang_loader

PREDICATE_NAMES = {"get_class_predicate", "_post_promo_predicate"}


def _collect_predicate_defs():
    tree = ast.parse(inspect.getsource(jang_loader))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in PREDICATE_NAMES
    ]


def _returns_only_false(stmt: ast.stmt) -> bool:
    """True if every Return inside stmt returns a falsy constant."""
    for node in ast.walk(stmt):
        if isinstance(node, ast.Return):
            val = node.value
            if val is None:
                continue
            if isinstance(val, ast.Constant) and not val.value:
                continue
            return False
    return True


def _has_to_quantized_floor(fn: ast.FunctionDef) -> bool:
    """The hasattr(m, "to_quantized") check must appear before any statement
    that can return a truthy value (True or a per-module override dict)."""
    for stmt in fn.body:
        dumped = ast.dump(stmt)
        if "hasattr" in dumped and "to_quantized" in dumped:
            return True
        if not _returns_only_false(stmt):
            return False
    return False


def test_all_quantize_predicates_floor_on_to_quantized():
    defs = _collect_predicate_defs()
    # 3 known sites: _post_promo_predicate (model_type promotion re-quantize),
    # fast-path VLM get_class_predicate, legacy v1 VLM get_class_predicate.
    assert len(defs) >= 3, (
        f"expected at least 3 quantize predicate closures in jang_loader, "
        f"found {len(defs)} — if a predicate was renamed, update "
        f"PREDICATE_NAMES so the to_quantized floor stays locked in"
    )
    unguarded = [fn.name for fn in defs if not _has_to_quantized_floor(fn)]
    assert not unguarded, (
        f"quantize predicates missing the hasattr(m, 'to_quantized') floor "
        f"before any truthy return: {unguarded} (vmlx#246 — an unguarded "
        f"predicate aborts the whole model load on router/gate modules)"
    )
