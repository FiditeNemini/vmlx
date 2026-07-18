"""ENGINE-AFFINE-JIT-DEFAULT-HYGIENE regression.

The JANG-affine JIT auto-default in cli.py must be gated on the same
mx.compile-safety conditions the runtime enforces: never claim JIT ON for
MLLM (mlx-vlm streaming) or hybrid SSM/Mamba cache models the runtime will
refuse to compile anyway (live-caught on Bonsai-27b-1bit-JANG: cli logged
"defaulting --enable-jit ON" and the runtime then logged "Skipping
mx.compile - MLLM hybrid cache").
"""

from pathlib import Path

import re

CLI = Path(__file__).resolve().parent.parent / "vmlx_engine" / "cli.py"


def test_affine_jit_default_gated_on_compile_safety() -> None:
    source = CLI.read_text()
    assert "_compile_unsafe = bool(" in source
    assert 'getattr(_mc, "is_mllm", False)' in source
    assert '"hybrid"' in source
    # The ON branch must require compile safety.
    assert (
        "if _is_affine and not _excluded_family and not _compile_unsafe:"
        in source
    )
    # The unsafe branch must log the truthful stay-OFF reason, not claim ON.
    assert "JIT default stays OFF" in source
    # The old unconditional branch must be gone.
    assert not re.search(
        r"if _is_affine and not _excluded_family:\s*\n\s*args\.enable_jit = True",
        source,
    )
