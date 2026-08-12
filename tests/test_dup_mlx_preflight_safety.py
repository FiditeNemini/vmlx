# SPDX-License-Identifier: Apache-2.0
"""The duplicate-mlx preflight must not mistake a plain folder for an install.

`sys.path` contains the CWD, so `os.path.join("", "mlx")` resolves to a RELATIVE
`mlx`. The preflight used a bare `os.path.isdir` on that, so on a machine whose
home directory holds an `~/mlx` folder of source checkouts — which is the layout
on the box this engine is developed on — running the engine from $HOME made that
folder look like a second MLX install. The preflight then aborted startup with
exit(2) and printed `rm -rf '<that folder>'`.

Two separate failures, both fixed here:
  1. detection must require the markers a real installed MLX has, and
  2. the message must never contain a delete command at all. A user reading it
     is by definition a user whose engine just refused to start, i.e. the exact
     moment they paste whatever it says into a terminal.

V1626-CAMPAIGN.md:22 promised both and neither had landed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_SRC = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")


def _preflight_source() -> str:
    start = CLI_SRC.index("MULTIPLE `mlx` package locations")
    head = CLI_SRC.rindex("def ", 0, start)
    tail = CLI_SRC.index("\ndef ", start)
    return CLI_SRC[head:tail]


def test_preflight_never_suggests_deleting_anything():
    body = _preflight_source()
    for danger in ("rm -rf", "rm -r ", "shutil.rmtree", "os.remove", "unlink"):
        assert danger not in body, (
            f"the dup-mlx preflight emits {danger!r}. With the CWD on sys.path "
            "this message has named a user's entire source directory before"
        )


def test_detection_requires_real_package_markers_not_just_a_directory():
    body = _preflight_source()
    assert "_is_installed_mlx_package" in body, (
        "the preflight went back to treating any directory named mlx as an "
        "installed package"
    )
    # An empty sys.path entry means CWD; a relative match would be reported to
    # the user as a bare "mlx".
    assert "os.path.abspath" in body, (
        "candidate paths are no longer absolutised, so a relative CWD match can "
        "be printed as a bare 'mlx'"
    )
    assert not re.search(r"if os\.path\.isdir\(cand\) and cand not in", body), (
        "the bare isdir candidate check is back"
    )


def test_the_marker_check_actually_rejects_a_plain_directory(tmp_path):
    """Behavioural check on the real predicate, not just its source text."""
    import ast

    tree = ast.parse(CLI_SRC)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_is_installed_mlx_package"
    )
    namespace: dict = {}
    exec(compile(ast.Module([fn], []), "<preflight>", "exec"), {"os": __import__("os")}, namespace)
    is_pkg = namespace["_is_installed_mlx_package"]

    plain = tmp_path / "mlx"
    plain.mkdir()
    assert not is_pkg(str(plain)), "an empty folder named mlx is not an install"

    (plain / "some-checkout").mkdir()
    assert not is_pkg(str(plain)), (
        "a folder named mlx holding source checkouts — the exact ~/mlx layout "
        "that triggered this — must not match"
    )

    (plain / "__init__.py").write_text("")
    assert not is_pkg(str(plain)), "an __init__.py alone is not MLX"

    (plain / "core.cpython-313-darwin.so").write_text("")
    assert is_pkg(str(plain)), "a real MLX install must still be detected"


# --- Behavioural coverage for the predicate itself -------------------------
#
# The assertion removed above required the string "__init__.py" to appear in
# the preflight. That pinned the MECHANISM rather than the behaviour, and it
# pinned the WRONG mechanism: MLX ships `mlx` as a NAMESPACE package
# (`importlib.util.find_spec("mlx").loader is None`, no root `__init__.py`), so
# demanding one rejected every genuine install. Detection then found zero
# locations, `len(unique) <= 1` returned early, and the entire preflight was a
# no-op — while this test stayed green and reported the opposite.
#
# These exercise the predicate against real directory layouts instead, so the
# only way to pass is to actually classify them correctly.

import importlib.util

from vmlx_engine.cli import _is_installed_mlx_package


def test_the_real_installed_mlx_is_detected():
    """The regression that made the preflight inert.

    MLX is a namespace package. If this returns False, no duplicate can ever
    be found and the nanobind duplicate-key crash comes back unannounced.
    """
    spec = importlib.util.find_spec("mlx")
    assert spec is not None, "mlx is not importable in this environment"
    locations = list(spec.submodule_search_locations)
    assert locations, "mlx has no submodule search locations"
    for location in locations:
        assert _is_installed_mlx_package(location), (
            f"the installed MLX at {location} is not recognised as an MLX "
            "package, so the duplicate-mlx preflight cannot fire at all"
        )


def test_a_source_checkout_named_mlx_is_not_an_install(tmp_path):
    """The original hazard: ~/mlx full of repo checkouts."""
    folder = tmp_path / "mlx"
    (folder / "vllm-mlx").mkdir(parents=True)
    (folder / "vllm-mlx-r20-dsv4-local").mkdir(parents=True)
    assert not _is_installed_mlx_package(str(folder))


def test_a_bare_core_directory_is_not_enough(tmp_path):
    """A compiled artifact is required, not merely a folder called `core`.

    Dropping the `__init__.py` requirement removes the guard that used to stop
    an arbitrary source tree from matching, so the compiled-extension check has
    to carry that weight on its own.
    """
    folder = tmp_path / "mlx"
    (folder / "core").mkdir(parents=True)
    assert not _is_installed_mlx_package(str(folder))


def test_a_compiled_core_extension_marks_an_install(tmp_path):
    folder = tmp_path / "mlx"
    folder.mkdir()
    (folder / "core.cpython-313-darwin.so").touch()
    assert _is_installed_mlx_package(str(folder))


def test_a_compiled_core_nested_under_core_also_counts(tmp_path):
    folder = tmp_path / "mlx"
    (folder / "core").mkdir(parents=True)
    (folder / "core" / "core.cpython-313-darwin.so").touch()
    assert _is_installed_mlx_package(str(folder))


def test_a_missing_path_is_not_an_install(tmp_path):
    assert not _is_installed_mlx_package(str(tmp_path / "does-not-exist"))
