"""Loader for Zyphra ZAYA text bundles."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

from ..models.zaya import register_mlx_lm_zaya
from ..utils.quant_shape_inference import infer_quant_overrides_for_bundle

logger = logging.getLogger(__name__)


def _zaya_tokenizer_config(config: dict):
    """Build a Transformers config without mutating the model bundle.

    Older ZAYA bundles use ``rope_scaling: false`` to mean "no RoPE
    scaling". Transformers 5 models that field as ``dict | None`` and rejects
    the legacy boolean before AutoTokenizer can load. Normalize only that
    semantically equivalent legacy value in memory and pass the resulting
    config directly to AutoTokenizer through mlx-lm.
    """

    normalized = dict(config)
    changed = False
    for key in ("rope_scaling", "rope_parameters"):
        if normalized.get(key) is False:
            normalized[key] = None
            changed = True
    if not changed:
        return None

    from transformers.models.zaya.configuration_zaya import ZayaConfig

    logger.info(
        "ZAYA tokenizer compatibility: normalized legacy false RoPE config to None"
    )
    return ZayaConfig(**normalized)


def load_zaya_model(model_path: str | Path, *, lazy: bool = False):
    """Load a ZAYA BF16/MXFP4/affine bundle with the local CCA runtime.

    JANGTQ/MXTQ ZAYA bundles should still route through ``load_jang_model`` so
    jang_tools can replace ``switch_mlp`` projections with TurboQuant modules.
    """
    path = Path(model_path)
    register_mlx_lm_zaya()

    cfg = json.loads((path / "config.json").read_text())
    try:
        cfg = infer_quant_overrides_for_bundle(path, cfg)
    except Exception as exc:
        logger.debug("ZAYA quant-shape inference skipped: %s", exc)

    model, loaded_cfg = load_model(
        path,
        model_config=cfg,
        lazy=lazy,
        strict=True,
    )
    tokenizer_config = _zaya_tokenizer_config(cfg)
    tokenizer = load_tokenizer(
        path,
        tokenizer_config_extra=(
            {"config": tokenizer_config} if tokenizer_config is not None else None
        ),
        eos_token_ids=loaded_cfg.get("eos_token_id"),
    )
    if not hasattr(model, "config"):
        model.config = loaded_cfg
    if not lazy:
        mx.eval(model.parameters())
    logger.info(
        "ZAYA runtime loaded: layers=%s, cache=CCA(KV+conv_state+prev_hs), "
        "prefix/paged/L2 disabled until restore tests pass",
        len(getattr(model, "layers", [])),
    )
    return model, tokenizer
