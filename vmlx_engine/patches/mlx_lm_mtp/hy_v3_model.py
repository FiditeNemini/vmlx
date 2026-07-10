# SPDX-License-Identifier: Apache-2.0
"""Hy3 (hy_v3) native-MTP attach patch.

Unlike qwen35_model.py / deepseek_v4_model.py, all Hy3 model-side MTP code
(``Hy3MTPLayer``, ``mtp_forward``, ``make_mtp_cache``, ``attach_mtp``, the
``return_hidden`` ``__call__`` and the mtp-aware ``sanitize``) lives in
``jang_tools.hy3.model`` — the same module that registers itself as
``mlx_lm.models.hy_v3`` and supplies the base runtime model for every Hy3
JANG bundle. This patch only bridges vMLX's activation gate: it wraps
``Model.__init__`` to attach the MTP head when ``is_mtp_active()`` is set and
the config carries ``num_nextn_predict_layers > 0``.

Head design (mirrors vLLM HYV3MultiTokenPredictorLayer / DeepSeek-V3 MTP):
``eh_proj(concat([enorm(embed(next_ids)), hnorm(prenorm_hidden)]))`` -> one
full Hy3 MoE decoder layer -> ``final_layernorm`` -> shared fp32 lm_head.
The JANG converter ships those weights as ``mtp.0.*`` (final param names),
which is also what native_mtp.py's ``_MTP_LAYER_PATTERNS`` detection expects.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def apply() -> bool:
    """Idempotent. Returns True when the hy_v3 attach gate is in place (or the
    base module is unavailable — absence is not an error: the loader imports
    ``jang_tools.hy3`` lazily and non-Hy3 sessions never need this patch)."""
    try:
        from jang_tools.hy3 import register_mlx_lm_hy3

        register_mlx_lm_hy3()
    except Exception as exc:  # jang package absent/old — skip cleanly
        logger.debug("hy_v3 MTP patch skipped (jang_tools.hy3 unavailable: %s)", exc)
        return True

    hy = sys.modules.get("mlx_lm.models.hy_v3")
    if hy is None or not hasattr(hy, "Model"):
        logger.debug("hy_v3 MTP patch skipped (mlx_lm.models.hy_v3 not registered)")
        return True

    cls = hy.Model
    if not hasattr(cls, "attach_mtp"):
        logger.debug(
            "hy_v3 MTP patch skipped (jang_tools.hy3.model predates native MTP "
            "— upgrade jang for Hy3 speculative decode)"
        )
        return True
    if "_omlx_mtp_patched" in cls.__dict__:
        return True

    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        # Only attach the head when vMLX opted this load into MTP. With the
        # flag off the model has no ``mtp`` attribute: sanitize strips
        # ``mtp.*`` weights and BatchGenerator's _is_mtp_eligible bails out.
        from . import is_mtp_active

        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        if n_mtp > 0 and is_mtp_active():
            self.attach_mtp()

    cls.__init__ = __init__
    cls._omlx_mtp_patched = True
    logger.info("hy_v3 native-MTP attach gate applied")
    return True
