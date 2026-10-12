"""MiniCPM-V-4.7 (``minicpmv4_7``): vision tower + merger + Qwen3.5 text model.

Port of ``MiniCPMV4_7Model`` / ``MiniCPMV4_7ForConditionalGeneration``
(transformers 5.19.0 ``modeling_minicpmv4_7.py``):
  * ``get_image_features`` L610-638: ``use_vit_merger = downsample_mode != "4x"``;
    in 16x the merger sees ``target_sizes // 2``.
  * ``get_video_features`` L766-794: the same path on the frame strip.
  * ``forward`` L704-760: masked_scatter of image features at 248056 and video
    features at 248057 (count-checked), then canvas positions over the whole
    prompt (``compute_3d_position_ids``).
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import InputEmbeddingsFeatures

from .canvas import crop_token_runs, merge_factor_for
from .config import ModelConfig, TextConfig, text_model_is_moe
from .language import language_model_class
from .vision import Merger, VisionModel

logger = logging.getLogger("vmlx_engine")

_PER_LAYER_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_ZERO_CENTERED_NORM_SUFFIXES = _PER_LAYER_NORM_SUFFIXES + ("language_model.model.norm.weight",)


def _np_int(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value).astype(np.int64)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config.text_config, dict):
            config.text_config = TextConfig.from_dict(config.text_config)
        # mlx-vlm replaces vision_config after ModelConfig.__post_init__, so
        # re-apply the HF __post_init__ copy of the top-level insert layer.
        config.vision_config.insert_layer_id = int(config.insert_layer_id)
        self.config = config
        self.model_type = config.model_type
        text_config = config.text_config
        self.vision_tower = VisionModel(config.vision_config)
        self.merger = Merger(
            config.vision_config.hidden_size,
            text_config.hidden_size,
            merger_times=config.merger_times,
            merge_kernel_size=config.merge_kernel_size,
        )
        self.language_model = language_model_class(text_config)(text_config, config)

    # ------------------------------------------------------------------ media
    @property
    def layers(self):
        return self.language_model.model.layers

    def _vision_dtype(self):
        return self.vision_tower.embeddings.patch_embedding.weight.dtype

    def encode_media(
        self,
        pixel_values: mx.array,
        target_sizes,
        downsample_mode: Optional[str] = None,
        capture: Optional[dict] = None,
    ) -> mx.array:
        """Vision tower + merger for one NaViT strip: ``(sum tokens, llm_hidden)``."""
        mode = downsample_mode or self.config.downsample_mode
        if mode not in ("16x", "4x"):
            raise ValueError(f"MiniCPM-V-4.7 downsample_mode must be '16x' or '4x', got {mode!r}")
        sizes = _np_int(target_sizes).reshape(-1, 2)
        if pixel_values.ndim == 4 and pixel_values.shape[0] > 1:
            # The processor packs every crop into batch row 0 (HF uses [:1]).
            pixel_values = pixel_values[:1]
        hidden, merged_sizes = self.vision_tower(
            pixel_values.astype(self._vision_dtype()),
            sizes,
            use_vit_merger=(mode != "4x"),
            capture=capture,
        )
        features = self.merger(hidden, merged_sizes)
        if capture is not None:
            capture["vision_out"] = hidden
            capture["merger_out"] = features
        return features

    @staticmethod
    def _check_crop_tokens(input_ids: np.ndarray, token_id: int, sizes: np.ndarray, mode: str, modality: str):
        """Every crop run must hold h*w/divisor pads (processor/model mode agreement)."""
        divisor = merge_factor_for(mode) ** 2
        expected = [int(h) * int(w) // divisor for h, w in sizes.reshape(-1, 2).tolist()]
        runs = []
        for row in np.asarray(input_ids).reshape(-1, input_ids.shape[-1]):
            types = (row == token_id).astype(np.int64)
            runs.extend(end - start for _, start, end in crop_token_runs(types.tolist()))
        if runs != expected:
            raise ValueError(
                f"MiniCPM-V-4.7 {modality} placeholders do not match the {mode} crop grid: "
                f"runs={runs[:8]}{'...' if len(runs) > 8 else ''} expected={expected[:8]}"
                f"{'...' if len(expected) > 8 else ''}. The processor and the model must use the "
                "same downsample_mode."
            )

    @staticmethod
    def _scatter(embeds: mx.array, input_ids: np.ndarray, token_id: int, features: mx.array, modality: str):
        positions = np.flatnonzero(np.asarray(input_ids).reshape(-1) == int(token_id))
        if positions.size != int(features.shape[0]):
            raise ValueError(
                f"MiniCPM-V-4.7 {modality} features and tokens do not match: "
                f"tokens={positions.size}, features={int(features.shape[0])}"
            )
        batch, length, dim = embeds.shape
        flat = embeds.reshape(batch * length, dim)
        flat[mx.array(positions.astype(np.uint32))] = features.astype(embeds.dtype)
        return flat.reshape(batch, length, dim)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        **kwargs,
    ):
        # ``pixel_values_videos`` is a NAMED parameter: the batched engine picks
        # the video kwarg spelling by signature (_video_pixel_values_kwarg_name).
        lm = self.language_model
        cfg = self.config
        embeds = lm.model.embed_tokens(input_ids)
        ids = np.asarray(input_ids)
        has_image_pads = bool((ids == cfg.image_token_id).any())
        has_video_pads = bool((ids == cfg.video_token_id).any())

        if pixel_values is None and pixel_values_videos is None:
            if has_image_pads or has_video_pads:
                raise ValueError(
                    "MiniCPM-V-4.7 prompt holds media placeholders but no pixel values were given"
                )
            lm.reset_position_state()
            return InputEmbeddingsFeatures(inputs_embeds=embeds)

        mode = kwargs.get("downsample_mode") or cfg.downsample_mode
        target_sizes = kwargs.get("target_sizes")
        target_sizes_videos = kwargs.get("target_sizes_videos")
        if pixel_values is not None:
            if target_sizes is None:
                raise ValueError("MiniCPM-V-4.7 image pixels need target_sizes from the processor")
            sizes = _np_int(target_sizes)
            self._check_crop_tokens(ids, cfg.image_token_id, sizes, mode, "image")
            features = self.encode_media(pixel_values, sizes, mode)
            embeds = self._scatter(embeds, ids, cfg.image_token_id, features, "image")
        elif has_image_pads:
            raise ValueError("MiniCPM-V-4.7 prompt holds image placeholders but no image pixels")
        if pixel_values_videos is not None:
            if target_sizes_videos is None:
                raise ValueError("MiniCPM-V-4.7 video pixels need target_sizes_videos from the processor")
            sizes_v = _np_int(target_sizes_videos)
            self._check_crop_tokens(ids, cfg.video_token_id, sizes_v, mode, "video")
            features_v = self.encode_media(pixel_values_videos, sizes_v, mode)
            embeds = self._scatter(embeds, ids, cfg.video_token_id, features_v, "video")
        elif has_video_pads:
            raise ValueError("MiniCPM-V-4.7 prompt holds video placeholders but no video pixels")

        # Canvas positions for the WHOLE prompt, before any chunked prefill.
        positions, deltas = lm.get_rope_index(
            input_ids,
            attention_mask=kwargs.get("mask"),
            mm_token_type_ids=kwargs.get("mm_token_type_ids"),
            target_sizes=target_sizes,
            target_sizes_videos=target_sizes_videos,
            downsample_mode=mode,
        )
        lm._position_ids = positions
        lm._rope_deltas = deltas
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        pixel_values_videos: Optional[mx.array] = None,
        **kwargs,
    ):
        features = self.get_input_embeddings(
            input_ids, pixel_values, pixel_values_videos=pixel_values_videos, mask=mask, **kwargs
        )
        lm_kwargs = {}
        if kwargs.get("position_ids") is not None:
            lm_kwargs["position_ids"] = kwargs["position_ids"]
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            cache=cache,
            **lm_kwargs,
        )

    # ---------------------------------------------------------------- weights
    def sanitize(self, weights):
        """HF checkpoint names/layouts -> this module tree.

        * ``model.vision_tower.*`` -> ``vision_tower.*``; ``model.merger.*`` ->
          ``merger.*``; ``model.language_model.*`` -> ``language_model.model.*``;
          ``lm_head.*`` -> ``language_model.lm_head.*``.
        * patch embedding conv ``(D, C, P, P)`` -> MLX ``(D, P, P, C)``.
        * GDN depthwise conv ``(C, 1, K)`` -> MLX ``(C, K, 1)``.
        * stacked experts ``experts.gate_up_proj [E, 2I, H]`` (rows ``[:I]`` gate,
          ``[I:]`` up) and ``experts.down_proj [E, H, I]`` ->
          ``switch_mlp.{gate,up,down}_proj.weight``.
        * zero-centred RMSNorm weights stay zero-centred (``ZeroCenteredRMSNorm``
          adds the 1 in float32); a bundle that already stores ``1 + w`` is
          converted back exactly in float32.
        """
        text_cfg = self.config.text_config
        vision_cfg = self.config.vision_config
        weights = {k: v for k, v in weights.items() if not k.startswith("mtp.") and ".mtp." not in k}
        if getattr(text_cfg, "tie_word_embeddings", False) or self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        renamed = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key.startswith("model.vision_tower."):
                key = "vision_tower." + key[len("model.vision_tower.") :]
            elif key.startswith("model.merger."):
                key = "merger." + key[len("model.merger.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            renamed[key] = value

        out = {}
        moe = text_model_is_moe(text_cfg)
        for key, value in renamed.items():
            if key.endswith("embeddings.patch_embedding.weight") and value.ndim == 4:
                if value.shape[-1] != vision_cfg.num_channels and value.shape[1] == vision_cfg.num_channels:
                    value = value.transpose(0, 2, 3, 1)
            elif key.endswith("conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            elif moe and key.endswith(".mlp.experts.gate_up_proj"):
                prefix = key[: -len("experts.gate_up_proj")]
                intermediate = int(value.shape[-2]) // 2
                out[prefix + "switch_mlp.gate_proj.weight"] = value[:, :intermediate, :]
                out[prefix + "switch_mlp.up_proj.weight"] = value[:, intermediate:, :]
                continue
            elif moe and key.endswith(".mlp.experts.down_proj"):
                prefix = key[: -len("experts.down_proj")]
                out[prefix + "switch_mlp.down_proj.weight"] = value
                continue
            out[key] = value

        if self._norms_stored_shifted(weights, out):
            for key in list(out):
                if (
                    key.startswith("language_model.")
                    and key.endswith(_ZERO_CENTERED_NORM_SUFFIXES)
                    and out[key].ndim == 1
                ):
                    out[key] = out[key].astype(mx.float32) - 1.0
        return out

    def _norms_stored_shifted(self, raw_weights, live_weights) -> bool:
        """Storage contract: zero-centred ``w`` (the HF/source convention).

        Called per shard by the JANG loader, so it must not depend on which
        norms a shard happens to hold. HF-named keys are zero-centred by
        definition (transformers adds the 1 at runtime). For live-key shards
        a guard catches a ``1 + w`` export: per-layer norm means measured on
        both checkpoints span -0.263..0.741 zero-centred, so single tensors
        overlap a shifted export; the shard POPULATION mean separates them
        (zero-centred ~0.1-0.4, shifted ~1.1-1.4). The final norm (35B: alone
        with 5 others in the last shard; 1B mean 3.43, 35B 1.65) follows the
        decision of earlier shards.
        """
        if any(k.startswith("model.language_model.") for k in raw_weights):
            return False
        per_layer = [
            float(mx.mean(v.astype(mx.float32)).item())
            for k, v in live_weights.items()
            if k.startswith("language_model.")
            and k.endswith(_PER_LAYER_NORM_SUFFIXES)
            and v.ndim == 1
        ]
        if per_layer:
            shifted = float(np.mean(per_layer)) > 0.85
            if shifted:
                logger.warning(
                    "minicpmv4_7: RMSNorm weights are stored shifted (1 + w; shard "
                    "mean %.3f over %d norms); converting back to zero-centred "
                    "form in float32",
                    float(np.mean(per_layer)),
                    len(per_layer),
                )
            self._shifted_norm_storage = shifted
            return shifted
        return bool(getattr(self, "_shifted_norm_storage", False))

    # ------------------------------------------------------------ runtime hooks
    def prepare_acceleration(self) -> dict:
        """Group packed GDN projections of quantized bundles (no-op for BF16)."""
        from mlx_vlm.models.qwen3_5.language import prepare_quantized_projection_groups

        return prepare_quantized_projection_groups(self.language_model)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
