# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer top-level VLM.

Module names mirror the checkpoint prefixes exactly, so weights load without a
remap:

    language_model.model.*            -> self.language_model.model.*
    language_model.lm_head            -> self.language_model.lm_head
    model.vision_tower.*              -> self.vision_tower.*
    model.vision_adapter.fc1 / fc2    -> self.vision_adapter.fc1 / fc2
    model.vision_projection           -> self.vision_projection

The leading ``model.`` on the vision keys is stripped in ``sanitize`` rather
than by nesting an extra module, which would also have shifted the
language_model keys.
"""

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import LanguageModel
from .vision import MuseVisionAdapter, VisionModel


def _as_grid_list(grid_thw: Any) -> List[tuple]:
    """Normalize a grid_thw into a list of (t, h, w) int triples.

    Callers hand this over as an mx.array, a numpy array, a list of triples or
    a single flat triple depending on which surface they came through, and a
    silently mis-parsed grid scrambles every downstream index.
    """
    if grid_thw is None:
        return []
    if hasattr(grid_thw, "tolist"):
        grid_thw = grid_thw.tolist()
    if isinstance(grid_thw, (list, tuple)):
        if not grid_thw:
            return []
        first = grid_thw[0]
        if isinstance(first, (list, tuple)):
            return [tuple(int(v) for v in row) for row in grid_thw]
        if len(grid_thw) == 3:
            return [tuple(int(v) for v in grid_thw)]
    raise ValueError(f"Muse Glimmer: unrecognised grid_thw {grid_thw!r}")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        if config.vision_config is not None:
            self.vision_tower = VisionModel(config.vision_config)
            self.vision_adapter = MuseVisionAdapter(config)
            self.vision_projection = nn.Linear(
                int(config.projector_hidden_size),
                config.text_config.hidden_size,
                bias=False,
            )
        else:
            self.vision_tower = None
            self.vision_adapter = None
            self.vision_projection = None

    # ---- media -----------------------------------------------------------

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        image_grid_thw: Optional[Any] = None,
        video_grid_thw: Optional[Any] = None,
        **kwargs,
    ) -> mx.array:
        # Normed lookup (see MuseTextModel.embed). Media features are scattered
        # in AFTER the norm and stay raw, matching the reference's
        # masked_scatter over the normed embedding.
        embeds = self.language_model.model.embed(input_ids)
        if pixel_values is None or self.vision_tower is None:
            return embeds

        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if grid_thw is None:
            grid_thw = kwargs.get("grid_thw")
        grid_thw = _as_grid_list(grid_thw)
        if not grid_thw:
            raise ValueError(
                "Muse Glimmer received pixel_values without a grid_thw. Positions, "
                "rotary and the window partition are all defined over the patch "
                "grid, so it cannot be inferred from the pixel tensor alone."
            )

        features = self.vision_tower(pixel_values, grid_thw=grid_thw)
        features = self.vision_projection(
            self.vision_adapter(features, grid_thw=grid_thw)
        )
        return self._scatter_media(input_ids, embeds, features)

    def _scatter_media(
        self,
        input_ids: mx.array,
        embeds: mx.array,
        features: mx.array,
    ) -> mx.array:
        """Place projected media features at the image/video placeholders.

        Both placeholders share one feature stream in prompt order, so they are
        matched together rather than per-modality — a prompt that interleaves an
        image and a video would otherwise mis-assign the second one.
        """
        image_id = int(self.config.image_token_id)
        video_id = int(self.config.video_token_id)
        is_media = (input_ids == image_id) | (input_ids == video_id)
        count = int(mx.sum(is_media).item())
        if count == 0:
            return embeds

        flat = features.reshape(-1, features.shape[-1])
        if flat.shape[0] < count:
            raise ValueError(
                f"Muse Glimmer: {count} media placeholder(s) in the prompt but only "
                f"{flat.shape[0]} projected feature row(s); the processor and the "
                "prompt disagree about how many patches this media expands to."
            )
        mask = is_media[..., None]
        # Per-row cumsum restarts at 0 on every batch row, so with B > 1 every
        # row past the first re-read row 0's features. Offset each row by the
        # media tokens consumed before it. The engine co-batches VL requests,
        # so this is a live path, not a hypothetical.
        flags = is_media.astype(mx.int32)
        per_row = mx.sum(flags, axis=-1)
        row_offsets = mx.cumsum(per_row) - per_row  # exclusive prefix sum
        idx = mx.cumsum(flags, axis=-1) - 1 + row_offsets[..., None]
        gathered = flat[mx.clip(idx, 0, flat.shape[0] - 1)]
        return mx.where(mask, gathered, embeds)

    # ---- forward ---------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> mx.array:
        inputs_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            position_ids=kwargs.get("vision_position_ids"),
            image_grid_thw=kwargs.get("image_grid_thw"),
            video_grid_thw=kwargs.get("video_grid_thw"),
            grid_thw=kwargs.get("grid_thw"),
        )
        return self.language_model(
            inputs=None, inputs_embeds=inputs_embeds, mask=mask, cache=cache
        )

    # ---- weights ---------------------------------------------------------

    # Zero-centered RMSNorm gains in the text tower, matched by suffix because
    # the layer index varies. ALL FOUR of the decoder layer's norms are
    # centered, plus the final norm. The vision tower's norm1/norm2 are ordinary
    # LayerNorms and must NOT be shifted.
    _CENTERED_NORM_SUFFIXES = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    )

    @classmethod
    def _is_centered_norm_weight(cls, key: str) -> bool:
        if not key.startswith("language_model."):
            return False
        return key.endswith(cls._CENTERED_NORM_SUFFIXES) or key.endswith(
            "model.norm.weight"
        )

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Re-root the vision keys and fold the centered-norm ``+1`` in once.

        Two things happen here:

        * Only the vision side carries a leading ``model.``; ``language_model.*``
          is already at the right depth, so a blanket strip would break the text
          stack.
        * Every text-tower RMSNorm gain is stored ZERO-CENTERED, so the
          mathematical weight is ``1 + w``. Folding the ``+1`` in at load costs
          nothing per token; skipping it runs all 209 norms at gain ~0 instead of
          ~1, which crushes every sublayer's contribution to the residual stream.
          The model still loads and still emits text, which makes this the
          hardest class of port bug to spot -- it presents as degenerate
          repetition, not as a crash.
        """
        out: Dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("model.vision_tower.") or key.startswith(
                "model.vision_adapter."
            ) or key.startswith("model.vision_projection"):
                out[key[len("model.") :]] = value
            else:
                out[key] = value

        for key, value in out.items():
            if self._is_centered_norm_weight(key):
                out[key] = value + 1
        return out

    @property
    def layers(self):
        return self.language_model.model.layers
