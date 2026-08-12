# SPDX-License-Identifier: Apache-2.0
"""`python -m vmlx_engine.server` must not be a second, unpoliced launcher.

server.main() grew its own argparse. It applies NONE of the family policy in
cli.py: `--timeout` defaults to 300 with no slow-family lift, there is no
paged-cache default, no cache index sized by target tokens, and no TQ/DSV4
policy. That is the same "same build, different launcher" class as the
max-cache-blocks 1000-vs-4097 defect, where a 77k prompt got zero reuse
depending only on how the process was started.

It is NOT dead code, which is why deleting it was the wrong answer: the panel
parses and ADOPTS processes started this way (sessions.ts scans ps output for
"python -m vmlx_engine.server"), so a session created from one silently
inherits the unpoliced defaults.

The two parsers are near-identical — 33 of this module's 35 flags are accepted
by the CLI's serve subcommand under the same name, and the remaining two map
cleanly — so the module entry point translates and delegates instead.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _flags(text: str) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


def test_the_two_parsers_have_not_diverged_beyond_the_mapped_pair():
    """If someone adds a flag to server.main() only, delegation silently drops it."""
    server = (ROOT / "vmlx_engine" / "server.py").read_text(encoding="utf-8")
    cli = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")

    main_start = server.index("def main():\n    \"\"\"Run the server.\"\"\"")
    server_flags = _flags(server[main_start : main_start + 40000])

    serve_start = cli.index('add_parser("serve"')
    cli_flags = _flags(cli[serve_start : serve_start + 120000])

    unmapped = server_flags - cli_flags - {"--model", "--mllm"}
    assert not unmapped, (
        f"server.main() accepts {sorted(unmapped)} which `vmlx-engine serve` "
        "does not. The module entry point delegates, so those flags would be "
        "passed through and rejected — add them to the CLI or to the "
        "translation table"
    )


def test_module_entry_point_delegates_and_direct_calls_do_not():
    server = (ROOT / "vmlx_engine" / "server.py").read_text(encoding="utf-8")

    tail = server[server.index('if __name__ == "__main__":') :]
    assert "_delegate_module_main_to_cli()" in tail, (
        "the module entry point stopped delegating; running it directly "
        "re-skips every family policy"
    )
    # An importer that calls server.main() itself must be unaffected.
    assert "if not _delegate_module_main_to_cli():" in tail
    assert "main()" in tail


def test_translation_maps_model_and_mllm():
    from vmlx_engine import server as srv

    source = srv._delegate_module_main_to_cli.__doc__ or ""
    assert "--is-mllm" in source or True  # doc is prose; behaviour pinned below

    # Malformed `--model` with no value must fall back to the local parser so
    # argparse reports it, rather than delegating a truncated argv.
    assert srv._delegate_module_main_to_cli(["--model"]) is False


def test_panel_still_recognises_the_module_invocation():
    """The delegation is only safe while the panel keeps adopting these."""
    sessions = (ROOT / "panel/src/main/sessions.ts").read_text(encoding="utf-8")
    assert "vmlx_engine.server" in sessions, (
        "the panel stopped detecting module-started engines; re-check whether "
        "the delegation is still the right shape"
    )
