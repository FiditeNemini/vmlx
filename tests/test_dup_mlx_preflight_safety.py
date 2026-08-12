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
    assert "__init__.py" in body, (
        "detection no longer requires a package __init__.py, so a plain folder "
        "named mlx matches again"
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
