# SPDX-License-Identifier: Apache-2.0
"""dots3_note outer model — omni wiring (vision + audio towers over the LM).

Scatter contract (04-RUNTIME-HANDOFF §6): ONE processor call is the source
of truth for placeholder expansion. Image AND video features both land at
``image_token_id`` (151660) — ``video_token_id`` (151680) is vestigial at
runtime because the processor expands video frames to ``<|imgpad|>``.
Audio lands at ``audio_token_id`` (151720). A media payload whose
placeholder count is zero means the modality was silently dropped upstream;
that raises here rather than letting the model confabulate.
"""

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        self.vocab_size = config.vocab_size

        if config.vision_config is not None:
            from .vision import VisionModel

            self.vision_encoder = VisionModel(config.vision_config)
        else:
            self.vision_encoder = None
        if config.audio_config is not None:
            from .audio import AudioModel

            self.audio_encoder = AudioModel(config.audio_config)
        else:
            self.audio_encoder = None

    # ---- media -----------------------------------------------------------

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[Any] = None,
        video_grid_thw: Optional[Any] = None,
        input_features: Optional[Any] = None,
        audio_chunk_sample_lens: Optional[Any] = None,
        audio_chunk_token_lens: Optional[Any] = None,
        audio_chunk_counts: Optional[Any] = None,
        **kwargs,
    ) -> mx.array:
        embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            if self.vision_encoder is None:
                raise ValueError(
                    "dots3_note received pixel_values but the vision tower "
                    "was not constructed"
                )
            grid = image_grid_thw if image_grid_thw is not None else video_grid_thw
            if grid is None:
                grid = kwargs.get("grid_thw")
            if grid is None:
                raise ValueError(
                    "dots3_note received pixel_values without a grid_thw"
                )
            features = self.vision_encoder(pixel_values, grid)
            embeds = self._scatter_at(
                input_ids,
                embeds,
                features,
                int(self.config.image_token_id),
                "image/video",
            )

        if input_features is not None:
            if self.audio_encoder is None:
                raise ValueError(
                    "dots3_note received audio features but the audio tower "
                    "was not constructed"
                )
            audio_embeds, _lens = self.audio_encoder(
                input_features,
                audio_chunk_sample_lens,
                audio_chunk_token_lens,
                audio_chunk_counts,
            )
            embeds = self._scatter_at(
                input_ids,
                embeds,
                audio_embeds,
                int(self.config.audio_token_id),
                "audio",
            )

        return embeds

    def _scatter_at(
        self,
        input_ids: mx.array,
        embeds: mx.array,
        features: mx.array,
        token_id: int,
        label: str,
    ) -> mx.array:
        is_media = input_ids == token_id
        count = int(mx.sum(is_media).item())
        flat = features.reshape(-1, features.shape[-1]).astype(embeds.dtype)
        if count != flat.shape[0]:
            # Zero placeholders with a live payload = the modality was
            # dropped upstream; a partial mismatch = processor/prompt
            # disagreement. Both confabulate silently if allowed through.
            raise ValueError(
                f"dots3_note {label} scatter mismatch: {count} placeholder "
                f"token(s) (id {token_id}) vs {flat.shape[0]} feature row(s)"
            )
        if count == 0:
            return embeds
        mask = is_media[..., None]
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
    ):
        inputs_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=kwargs.get("image_grid_thw"),
            video_grid_thw=kwargs.get("video_grid_thw"),
            input_features=kwargs.get("input_features"),
            audio_chunk_sample_lens=kwargs.get("audio_chunk_sample_lens"),
            audio_chunk_token_lens=kwargs.get("audio_chunk_token_lens"),
            audio_chunk_counts=kwargs.get("audio_chunk_counts"),
            grid_thw=kwargs.get("grid_thw"),
        )
        return self.language_model(
            inputs=None,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            return_hidden=kwargs.get("return_hidden", False),
        )

    # ---- weights ---------------------------------------------------------

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Re-root the checkpoint's top-level LM keys under language_model.

        The bundle stores ``model.*`` / ``lm_head.*`` at top level while the
        towers are already prefixed (``vision_encoder.*``,
        ``audio_encoder.*``). The MTP layer rides along untouched:
        ``model.layers.46.*`` and ``model.mtp.*`` land on the modules the
        language model builds for them.

        🚨 dots3 norm gains look zero-centered (means 0.002-0.49) but are
        NOT — they multiply as stored. The Muse-style ``+1`` fold does not
        apply here; adding it is the trap, not omitting it.
        """
        out: Dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("model.") or key.startswith("lm_head."):
                out["language_model." + key] = value
            else:
                out[key] = value
        return out

    @property
    def layers(self):
        return self.language_model.layers
