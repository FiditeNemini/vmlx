"""Identify the measured MiMo mixed-format bundle without loading a runtime.

Preserved media tensors do not establish that a serving adapter is available.
In particular, capability discovery must not import the legacy MiMo adapter:
that registration also patches shared mlx-lm classes process-wide.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_mimo_v26_contract(bundle_path: str | Path | None) -> dict | None:
    if not bundle_path:
        return None
    path = Path(bundle_path)
    try:
        cfg = json.loads((path / "config.json").read_text())
        if cfg.get("model_type") != "mimo_v2":
            return None
        meta = json.loads((path / "jang_config.json").read_text())
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(meta, dict) or meta.get("weight_format") != "mixed_affine_mxfp4":
        return None
    return meta


def mimo_v26_media_requested(contract: dict) -> bool:
    caps = contract.get("capabilities") or {}
    modalities = caps.get("modalities") or {}
    return any(
        contract.get(f"has_{name}") is True or modalities.get(name) is True
        for name in ("vision", "video", "audio")
    )


def mimo_v26_media_enabled(contract: dict) -> bool:
    # Qualification uses an explicit process-local opt-in; bundle capability
    # stamps remain false until real image/video/audio generation is verified.
    return mimo_v26_media_requested(contract) or os.environ.get("VMLX_MIMO26_MEDIA_TEST") == "1"


def mimo_v26_modalities(bundle_path, contract):
    if not mimo_v26_media_enabled(contract):
        return ["text"]
    path = Path(bundle_path)
    cfg = json.loads((path / "config.json").read_text())
    weights = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    result = ["text"]
    if cfg.get("vision_config") and any(k.startswith("visual.") for k in weights):
        if cfg.get("image_token_id") is not None:
            result.append("vision")
        if cfg.get("video_token_id") is not None:
            result.append("video")
    if (cfg.get("audio_config") and (path / "audio_tokenizer/model.safetensors").is_file()
            and any(k.startswith("audio_encoder.") for k in weights)
            and any(k.startswith("speech_embeddings.") for k in weights)):
        result.append("audio")
    return result
