from dataclasses import replace

from mlx_vlm.models.qwen3_vl import VisionModel as Qwen3VLVisionModel


class VisionModel(Qwen3VLVisionModel):
    """Qwen4-Exp retains the Qwen3-VL image/video tower contract."""

    def __init__(self, config):
        # mlx-vlm's Qwen3-VL tower has no architecture branch on model_type,
        # but its constructor rejects names outside the older Qwen3 allowlist.
        # Build the identical tower through the supported base identity, then
        # restore the real bundle identity for diagnostics and serialization.
        base_config = replace(config, model_type="qwen3_vl")
        super().__init__(base_config)
        self.config = config
        self.model_type = config.model_type
