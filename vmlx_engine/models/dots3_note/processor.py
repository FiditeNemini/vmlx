# SPDX-License-Identifier: Apache-2.0
"""dots3_note image/video/audio preprocessing.

Ported from the transformers PR sources (processing/image_processing/
video_processing/feature_extraction_dots3_note.py), in the interface shape
mlx-vlm/vmlx calls (see ``muse_glimmer/processor.py``): the chat template is
rendered OUTSIDE; this class turns the rendered prompt + media into expanded
input_ids and tower-ready tensors.

Contracts that bite:

- 🚨 Video rides the IMAGE token path: the template's video placeholder
  (``<|video_pad|>`` 151680) is consumed HERE and replaced by N x
  ``<|imgpad|>`` (151660); ``config.video_token_id`` is vestigial at runtime
  and must never reach the model. An 8-frame 224x224 clip expands to 968
  image tokens (frames are upscaled to the 128-merged-block per-frame FLOOR).
- Every placeholder expansion is COUNT-CHECKED against the tensor rows the
  tower will emit. Zero placeholders while media was supplied raises — a
  silently dropped image/audio leaves the model confabulating a description
  of media it never received (the muse_glimmer failure mode).
- Frames are independent images: temporal_patch_size=1, no temporal conv, and
  each sampled video frame contributes its own ``image_grid_thw`` row [1,h,w].
- Video frames get the PR's training-consistent JPEG round trip (quality 85)
  before patchification — deliberate pixel change, not a compression bug.
- Audio: 128-bin log-mel, 60 s chunks @ 16 kHz, token count
  ceil(samples / 1280) per chunk (hop 160 x conv stride 8 x merge 1) — the
  same arithmetic the tower's conv stem realizes. Decoding file paths/bytes
  uses librosa (the house choice — see mllm_batch_generator).
"""

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# From the PR: Qwen2-VL preprocessing lineage.
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Release vision size (PR ``_RELEASE_VISION_SIZE``): NOT the Qwen2-VL defaults.
_IMAGE_MIN_PIXELS = 56 * 56
_IMAGE_MAX_PIXELS = (36 * 28) ** 2

# Native-video budget constants (video_processing_dots3_note.py).
_ALIGN = 28  # patch * merge; "patches" below are 28 px merged blocks
_MIN_FRAMES = 4
_PF_FLOOR = 128
_PF_CEIL = 1024
_FPS_CAP = 1.0
_FPS_MIN = 0.2
_FRAME_OVERHEAD = 15
_BUDGET_OVERHEAD = 2240
_DEFAULT_SEQUENCE_LENGTH = 524_288
# tokenizer-free fallback for the PR's tokenized prompt overhead; only binds
# for videos long enough to exhaust a ~393k-token visual budget.
_OVERHEAD_FALLBACK = 96

# Placeholder ids (mirrors config.py ModelConfig; constructor can override).
_IMAGE_TOKEN_ID = 151660  # <|imgpad|>
_VIDEO_TOKEN_ID = 151680  # <|video_pad|> — consumed here, never emitted
_AUDIO_TOKEN_ID = 151720  # <|audio_comp_pad|>
# Negative, so it can never collide with a real vocabulary id. Video frames are
# expanded into this first and converted to <|imgpad|> only after the image
# placeholders have been resolved — otherwise the image expansion counts the
# video's own expanded tokens as unexpanded placeholders.
_VIDEO_EXPANSION_SENTINEL = -991680


# --------------------------------------------------------------------- shared


def _as_pil(image):
    """Accept PIL / numpy / path / URL / bytes, like the muse processor."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        text = str(image)
        if text.startswith("data:"):
            payload = text.split(",", 1)[1] if "," in text else ""
            return Image.open(io.BytesIO(base64.b64decode(payload)))
        if text.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(text, timeout=60) as response:
                return Image.open(io.BytesIO(response.read()))
        return Image.open(text)
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))

    array = np.asarray(image)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.dtype != np.uint8:
        peak = float(array.max()) if array.size else 0.0
        if peak <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _convert_to_rgb(image):
    """PR behavior: RGBA composites onto WHITE, not a bare convert()."""
    from PIL import Image

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image if image.mode == "RGB" else image.convert("RGB")


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> Tuple[int, int]:
    """Verbatim from image_processing_dots3_note.py."""
    if min(height, width) < factor // 4:
        raise ValueError(
            f"Image height and width must be at least {factor // 4}, got {height}x{width}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must be smaller than 200")

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
        if resized_height * resized_width > max_pixels:
            scale = math.sqrt(resized_height * resized_width / max_pixels)
            resized_height = max(factor, math.floor(resized_height / scale / factor) * factor)
            resized_width = max(factor, math.floor(resized_width / scale / factor) * factor)
    return resized_height, resized_width


def merged_token_count(grid: Sequence[int], merge_size: int) -> int:
    t, h, w = (int(v) for v in grid)
    return (t * h * w) // (merge_size ** 2)


def expand_media_placeholders(
    input_ids: Sequence[int],
    placeholder_id: int,
    token_counts: Sequence[int],
    emit_id: Optional[int] = None,
    modality: str = "media",
) -> List[int]:
    """Replace occurrence i of ``placeholder_id`` with ``token_counts[i]``
    copies of ``emit_id`` (video passes emit_id=image token).

    Occurrence count MUST equal the media count: zero placeholders while media
    was supplied means the scatter would silently drop the media and the model
    would confabulate — raise instead.
    """
    emit = placeholder_id if emit_id is None else int(emit_id)
    ids = list(input_ids)
    positions = [i for i, t in enumerate(ids) if t == placeholder_id]
    if len(positions) != len(token_counts):
        raise ValueError(
            f"dots3_note: {len(positions)} {modality} placeholder(s) "
            f"(id {placeholder_id}) in the prompt but {len(token_counts)} "
            f"{modality} item(s) supplied; the prompt and the media disagree."
        )
    out: List[int] = []
    cursor = 0
    for position, count in zip(positions, token_counts):
        count = int(count)
        if count <= 0:
            raise ValueError(
                f"dots3_note: {modality} item expands to {count} tokens"
            )
        out.extend(ids[cursor:position])
        out.extend([emit] * count)
        cursor = position + 1
    out.extend(ids[cursor:])
    return out


def _patchify_frame(
    chw: np.ndarray, patch_size: int, merge_size: int
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """One [3, H, W] frame -> merge-block-major rows [gh*gw, C*p*p], grid (1,gh,gw).

    temporal_patch_size is 1 for dots3_note: every frame stands alone. Row
    layout is channel-major so the tower can reshape (-1, C, p, p) directly.
    """
    channels, height, width = chw.shape
    p, m = patch_size, merge_size
    grid_h, grid_w = height // p, width // p
    patches = chw.reshape(1, 1, channels, grid_h // m, m, p, grid_w // m, m, p)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, channels * p * p)
    return np.ascontiguousarray(flat, dtype=np.float32), (1, grid_h, grid_w)


# --------------------------------------------------------------------- images


class Dots3NoteImageProcessor:
    """Resize -> normalize -> patchify per the PR (patch 14, merge 2, T=1)."""

    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 1,
        min_pixels: int = _IMAGE_MIN_PIXELS,
        max_pixels: int = _IMAGE_MAX_PIXELS,
        image_mean: Sequence[float] = _OPENAI_CLIP_MEAN,
        image_std: Sequence[float] = _OPENAI_CLIP_STD,
        rescale_factor: float = 1.0 / 255.0,
        **_: Any,
    ) -> None:
        if temporal_patch_size != 1:
            # The PR forces this even when a legacy config says 2.
            raise ValueError("dots3_note vision uses temporal_patch_size=1")
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.temporal_patch_size = 1
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.rescale_factor = float(rescale_factor)

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def _to_chw(self, pil, size: Tuple[int, int]) -> np.ndarray:
        from PIL import Image

        pil = _convert_to_rgb(pil)
        if (pil.height, pil.width) != size:
            pil = pil.resize((size[1], size[0]), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(pil, dtype=np.float32) * self.rescale_factor
        arr = arr.transpose(2, 0, 1)
        return (arr - self.image_mean) / self.image_std

    def __call__(
        self, images: Sequence[Any]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        all_patches, grids = [], []
        for image in images:
            pil = _as_pil(image)
            size = smart_resize(
                pil.height,
                pil.width,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            patches, grid = _patchify_frame(
                self._to_chw(pil, size), self.patch_size, self.merge_size
            )
            all_patches.append(patches)
            grids.append(grid)
        if not all_patches:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.concatenate(all_patches, axis=0), grids


# --------------------------------------------------------------------- video


def _resolve_video_budget(
    tokenizer, sequence_length: Optional[int], output_reserve: Optional[int],
    max_new_tokens: int,
) -> Tuple[int, int]:
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max is None or tokenizer_max > 1e29:
        tokenizer_max = _DEFAULT_SEQUENCE_LENGTH
    if sequence_length is None:
        sequence_length = tokenizer_max
    elif sequence_length > tokenizer_max:
        raise ValueError(
            f"sequence_length must not exceed model_max_length ({tokenizer_max})"
        )
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if output_reserve is not None and output_reserve < 0:
        raise ValueError(f"output_reserve must be non-negative, got {output_reserve}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    effective_reserve = max(
        sequence_length // 4 if output_reserve is None else output_reserve,
        max_new_tokens,
    )
    if effective_reserve >= sequence_length:
        raise ValueError("output_reserve/max_new_tokens must leave room for video input")
    return sequence_length, effective_reserve


def _compute_target_size(
    orig_height: int, orig_width: int, min_pixels: int, max_pixels: int
) -> Tuple[int, int]:
    """Verbatim from video_processing_dots3_note.py (28 px alignment)."""
    height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    if height * width > max_pixels:
        scale = math.sqrt(orig_height * orig_width / max_pixels)
        height = max(_ALIGN, math.floor(orig_height / scale / _ALIGN) * _ALIGN)
        width = max(_ALIGN, math.floor(orig_width / scale / _ALIGN) * _ALIGN)
    elif height * width < min_pixels:
        scale = math.sqrt(min_pixels / max(1, orig_height * orig_width))
        height = math.ceil(orig_height * scale / _ALIGN) * _ALIGN
        width = math.ceil(orig_width * scale / _ALIGN) * _ALIGN
        if height * width > max_pixels:
            scale = math.sqrt(height * width / max_pixels)
            height = max(_ALIGN, math.floor(height / scale / _ALIGN) * _ALIGN)
            width = max(_ALIGN, math.floor(width / scale / _ALIGN) * _ALIGN)
    return int(height), int(width)


def _real_patches_at(orig_height: int, orig_width: int, patch_cap: int) -> int:
    height, width = _compute_target_size(
        orig_height,
        orig_width,
        _PF_FLOOR * _ALIGN * _ALIGN,
        max(_PF_FLOOR, patch_cap) * _ALIGN * _ALIGN,
    )
    return (height // _ALIGN) * (width // _ALIGN)


def _frame_hard_cap(sequence_length: int) -> int:
    required = max(1, (sequence_length - _BUDGET_OVERHEAD) // (_PF_FLOOR + _FRAME_OVERHEAD))
    if required <= 1024:
        return 1024
    cap = 1
    while cap < required:
        cap <<= 1
    return cap


def _solve_degrade(
    visual_budget: int,
    duration: float,
    orig_height: int,
    orig_width: int,
    orig_fps: float,
    sequence_length: int,
) -> Tuple[int, int]:
    """Verbatim from video_processing_dots3_note.py."""
    aligned_height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    aligned_width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    original_patch_cap = (aligned_height // _ALIGN) * (aligned_width // _ALIGN)
    fps_cap = min(_FPS_CAP, max(orig_fps, 1e-6))
    patch_cap = min(_PF_CEIL, max(original_patch_cap, _PF_FLOOR))
    frame_cap = _frame_hard_cap(sequence_length)

    def usage(scale: float) -> Tuple[int, int, int]:
        fps = _FPS_MIN + scale * (fps_cap - _FPS_MIN)
        candidate_patch_cap = _PF_FLOOR + scale * (patch_cap - _PF_FLOOR)
        num_frames = max(_MIN_FRAMES, min(int(round(duration * fps)), frame_cap))
        patches = _real_patches_at(orig_height, orig_width, int(round(candidate_patch_cap)))
        return num_frames * (patches + _FRAME_OVERHEAD), int(round(candidate_patch_cap)), num_frames

    if usage(1.0)[0] <= visual_budget:
        _, candidate_patch_cap, num_frames = usage(1.0)
        return num_frames, candidate_patch_cap

    floor_cost = _real_patches_at(orig_height, orig_width, _PF_FLOOR) + _FRAME_OVERHEAD
    if usage(0.0)[0] > visual_budget:
        return max(_MIN_FRAMES, min(visual_budget // floor_cost, frame_cap)), _PF_FLOOR

    low, high = 0.0, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        if usage(middle)[0] <= visual_budget:
            low = middle
        else:
            high = middle
    _, candidate_patch_cap, num_frames = usage(low)
    return num_frames, candidate_patch_cap


def _jpeg_roundtrip(image, quality: int):
    """PR training-consistent JPEG round trip — intentional pixel change."""
    from PIL import Image

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _as_frame_sequence(videos: Any) -> List[Any]:
    """Normalize the engine's video payload into a list of frames.

    The engine hands back a single 4-D (frames, H, W, C) ndarray, a list of
    frames, or a list wrapping one of those.
    """
    if videos is None:
        return []
    candidate = videos
    if isinstance(candidate, (list, tuple)) and len(candidate) == 1:
        candidate = candidate[0]
    ndim = getattr(candidate, "ndim", None)
    if ndim == 4:
        return [np.asarray(frame) for frame in candidate]
    if ndim == 3:
        return [np.asarray(candidate)]
    if isinstance(candidate, (list, tuple)):
        return list(candidate)
    return [candidate]


class Dots3NoteVideoProcessor(Dots3NoteImageProcessor):
    """Pre-decoded frame path (PR ``_prepare_decoded_frames`` semantics).

    Frames are sampled/sized under the PR's degrade solver, JPEG-roundtripped,
    then patchified as INDEPENDENT images. Small frames are UPSCALED to the
    128-merged-block per-frame floor: 8 frames of 224x224 -> 308x308 -> 121
    merged tokens each -> 968 image tokens total.

    Native container decoding (torchcodec, timestamped audio interleave) is
    NOT ported: vmlx hands decoded frames to this path.
    """

    def __call__(  # type: ignore[override]
        self,
        frames: Sequence[Any],
        fps: Optional[float] = None,
        tokenizer=None,
        sequence_length: Optional[int] = None,
        output_reserve: Optional[int] = None,
        max_new_tokens: int = 0,
        jpeg_quality: int = 85,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        from PIL import Image

        frames = list(frames)
        # The engine's video fallback hands per-video containers: frames may
        # arrive as [[frame, ...]] and fps as [fps] for a single video.
        if len(frames) == 1 and isinstance(frames[0], (list, tuple)):
            frames = list(frames[0])
        if isinstance(fps, (list, tuple)):
            fps = fps[0] if fps else None
        if hasattr(fps, "item"):
            fps = fps.item()
        if not frames:
            return np.zeros((0, 0), dtype=np.float32), []
        fps = max(float(fps or 1.0), 1e-6)  # decoded-frame metadata default
        duration = len(frames) / fps
        first = _as_pil(frames[0])
        orig_height, orig_width = first.height, first.width

        seq_len, reserve = _resolve_video_budget(
            tokenizer, sequence_length, output_reserve, max_new_tokens
        )
        input_length = seq_len - reserve
        if tokenizer is not None:
            overhead = (
                len(tokenizer.encode(
                    "<|system|>You are a helpful assistant.<|endofsystem|>\n",
                    add_special_tokens=False,
                ))
                + 2
                + len(tokenizer.encode("<video_0>", add_special_tokens=False))
                + 64
            )
        else:
            overhead = _OVERHEAD_FALLBACK
        visual_budget = max(_PF_FLOOR + _FRAME_OVERHEAD, input_length - overhead)

        num_frames, patch_cap = _solve_degrade(
            visual_budget, duration, orig_height, orig_width, fps, input_length
        )
        num_frames = min(max(1, num_frames), len(frames))
        indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        selected = sorted(set(indices.tolist()))
        target_height, target_width = _compute_target_size(
            orig_height,
            orig_width,
            _PF_FLOOR * _ALIGN * _ALIGN,
            patch_cap * _ALIGN * _ALIGN,
        )

        all_patches, grids = [], []
        for index in selected:
            image = _convert_to_rgb(_as_pil(frames[index]))
            if image.size != (target_width, target_height):
                image = image.resize(
                    (target_width, target_height), Image.Resampling.BICUBIC
                )
            image = _jpeg_roundtrip(image, jpeg_quality)
            arr = np.asarray(image, dtype=np.float32) * self.rescale_factor
            arr = (arr.transpose(2, 0, 1) - self.image_mean) / self.image_std
            patches, grid = _patchify_frame(arr, self.patch_size, self.merge_size)
            all_patches.append(patches)
            grids.append(grid)
        return np.concatenate(all_patches, axis=0), grids


# --------------------------------------------------------------------- audio


def _hz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        freq >= min_log_hz,
        min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep,
        mels,
    )


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = mels * f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        mels >= min_log_mel,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        freqs,
    )


def _mel_filter_bank(num_frequency_bins: int, num_mel_filters: int,
                     sampling_rate: int) -> np.ndarray:
    """Slaney-scale, slaney-normalized bank [n_mels, n_freqs] (librosa math,
    what transformers' mel_filter_bank produces for the PR extractor)."""
    fft_freqs = np.linspace(0, sampling_rate / 2, num_frequency_bins)
    mel_min = _hz_to_mel_slaney(np.array(0.0))
    mel_max = _hz_to_mel_slaney(np.array(sampling_rate / 2.0))
    mel_points = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    hz_points = _mel_to_hz_slaney(mel_points)
    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    bank = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney norm: each filter integrates to ~equal energy.
    enorm = 2.0 / (hz_points[2 : num_mel_filters + 2] - hz_points[:num_mel_filters])
    bank *= enorm[:, None]
    return bank.astype(np.float32)


def compute_audio_token_length(
    num_samples: int, *, chunk_samples: int = 960_000, token_stride: int = 1_280
) -> int:
    """Exact number of AE embeddings emitted for one waveform (PR verbatim)."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if chunk_samples <= 0 or token_stride <= 0:
        raise ValueError("chunk_samples and token_stride must be positive")
    return sum(
        math.ceil(min(chunk_samples, num_samples - start) / token_stride)
        for start in range(0, num_samples, chunk_samples)
    )


class Dots3NoteFeatureExtractor:
    """16 kHz mono waveform -> log-mel chunks + the tower's length metadata.

    numpy port of feature_extraction_dots3_note.py (torch.stft center=True,
    reflect pad, periodic hann, power-2 magnitudes with the LAST frame
    dropped, slaney mel, log10 clamped to an 8-decade dynamic range, then
    (x+4)/4). Output keys use the tower-facing spellings ``chunk_sample_lens``
    / ``chunk_token_lens`` (HF spells them *_lengths).
    """

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_seconds: int = 60,
        conv_temporal_stride: int = 8,  # three stride-2 convs in the tower
        merge_factor: int = 1,
        **_: Any,
    ) -> None:
        self.feature_size = int(feature_size)
        self.sampling_rate = int(sampling_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.chunk_seconds = int(chunk_seconds)
        self.chunk_samples = self.chunk_seconds * self.sampling_rate
        self.token_stride = self.hop_length * int(conv_temporal_stride) * int(merge_factor)
        self.mel_filters = _mel_filter_bank(
            1 + self.n_fft // 2, self.feature_size, self.sampling_rate
        )
        # torch.hann_window(periodic=True): denominator N, not N-1.
        n = np.arange(self.n_fft, dtype=np.float64)
        self._window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / self.n_fft)).astype(
            np.float64
        )

    def token_length(self, num_samples: int) -> int:
        return compute_audio_token_length(
            num_samples,
            chunk_samples=self.chunk_samples,
            token_stride=self.token_stride,
        )

    def _log_mel(self, waveform: np.ndarray) -> np.ndarray:
        x = waveform.astype(np.float64)
        pad = self.n_fft // 2
        x = np.pad(x, (pad, pad), mode="reflect")
        n_frames = 1 + (len(x) - self.n_fft) // self.hop_length
        idx = (
            np.arange(self.n_fft)[None, :]
            + self.hop_length * np.arange(n_frames)[:, None]
        )
        spec = np.fft.rfft(x[idx] * self._window[None, :], axis=-1)
        # torch path drops the LAST time frame (stft[..., :-1]).
        magnitudes = (np.abs(spec[:-1]) ** 2).T  # [n_freqs, frames]
        # Accelerate's BLAS sets spurious FP status flags on this matmul even
        # with fully finite operands (verified any layout/dtype); silence the
        # flags but keep a HARD finiteness check so real blowups stay loud.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mel = self.mel_filters.astype(np.float64) @ magnitudes
        if not np.isfinite(mel).all():
            raise ValueError("dots3_note: non-finite mel spectrogram")
        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        return (((log_spec + 4.0) / 4.0)).astype(np.float32)

    def __call__(
        self, raw_speech, sampling_rate: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"expected {self.sampling_rate} Hz audio, received {sampling_rate} Hz; "
                "resample before feature extraction"
            )
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim <= 2:
            clips = [raw_speech]
        elif isinstance(raw_speech, (list, tuple)) and raw_speech and isinstance(
            raw_speech[0], (int, float, np.integer, np.floating)
        ):
            clips = [np.asarray(raw_speech)]
        else:
            clips = [np.asarray(clip) for clip in raw_speech]

        chunks: List[np.ndarray] = []
        chunk_sample_lens: List[int] = []
        chunk_token_lens: List[int] = []
        audio_token_lengths: List[int] = []
        audio_chunk_counts: List[int] = []
        chunk_audio_indices: List[int] = []
        for audio_index, clip in enumerate(clips):
            waveform = np.asarray(clip, dtype=np.float32).squeeze()
            if waveform.ndim != 1:
                raise ValueError(
                    f"dots3_note audio must be mono, got shape={waveform.shape}"
                )
            if waveform.size == 0:
                raise ValueError("audio waveform must contain at least one sample")
            per_audio_tokens = 0
            per_audio_chunks = 0
            for start in range(0, waveform.size, self.chunk_samples):
                chunk = waveform[start : start + self.chunk_samples]
                sample_length = int(chunk.size)
                token_length = math.ceil(sample_length / self.token_stride)
                if sample_length < self.chunk_samples:
                    chunk = np.pad(chunk, (0, self.chunk_samples - sample_length))
                chunks.append(chunk)
                chunk_sample_lens.append(sample_length)
                chunk_token_lens.append(token_length)
                chunk_audio_indices.append(audio_index)
                per_audio_tokens += token_length
                per_audio_chunks += 1
            audio_token_lengths.append(per_audio_tokens)
            audio_chunk_counts.append(per_audio_chunks)

        input_features = np.stack([self._log_mel(chunk) for chunk in chunks])
        return {
            "input_features": input_features,
            "chunk_sample_lens": np.asarray(chunk_sample_lens, dtype=np.int64),
            "chunk_token_lens": np.asarray(chunk_token_lens, dtype=np.int64),
            "audio_token_lengths": np.asarray(audio_token_lengths, dtype=np.int64),
            "audio_chunk_counts": np.asarray(audio_chunk_counts, dtype=np.int64),
            "chunk_audio_indices": np.asarray(chunk_audio_indices, dtype=np.int64),
        }


# ------------------------------------------------------------------ processor


class Dots3NoteProcessor:
    """Tokenizer + media processors in the shape mlx-vlm/vmlx calls.

    vmlx renders the chat template BEFORE calling this; the job here is
    placeholder expansion + tower tensors. Everything not defined here falls
    through to the tokenizer (detokenizer, eos ids, stopping criteria).
    """

    def __init__(
        self,
        tokenizer,
        image_processor: Optional[Dots3NoteImageProcessor] = None,
        video_processor: Optional[Dots3NoteVideoProcessor] = None,
        feature_extractor: Optional[Dots3NoteFeatureExtractor] = None,
        image_token_id: int = _IMAGE_TOKEN_ID,
        video_token_id: int = _VIDEO_TOKEN_ID,
        audio_token_id: int = _AUDIO_TOKEN_ID,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor or Dots3NoteImageProcessor()
        self.video_processor = video_processor or Dots3NoteVideoProcessor()
        self.feature_extractor = feature_extractor or Dots3NoteFeatureExtractor()
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.audio_token_id = int(audio_token_id)

    # ---- delegation -------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.__dict__["tokenizer"], name)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # ---- audio decode (librosa is the house choice) -----------------------

    def _load_waveform(self, item) -> np.ndarray:
        sr = self.feature_extractor.sampling_rate
        if isinstance(item, (str, Path)):
            text = str(item)
            if text.startswith("data:"):
                payload = text.split(",", 1)[1] if "," in text else ""
                return self._load_waveform(base64.b64decode(payload))
            import librosa

            waveform, _ = librosa.load(text, sr=sr, mono=True)
            return np.asarray(waveform, dtype=np.float32)
        if isinstance(item, (bytes, bytearray)):
            import librosa

            waveform, _ = librosa.load(io.BytesIO(bytes(item)), sr=sr, mono=True)
            return np.asarray(waveform, dtype=np.float32)
        # Arrays are the caller's responsibility to already be 16 kHz mono.
        return np.asarray(item, dtype=np.float32)

    # ---- the call mlx-vlm makes ------------------------------------------

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        return_tensors=None,
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError("dots3_note processor handles one prompt at a time.")
            text = text[0]
        # A video and separate images CAN coexist — a multiturn chat that showed
        # a video on one turn and an image on a later turn sends both, because
        # media is collected across the whole message history. What must not
        # happen is the two branches overwriting each other's `pixel_values`:
        # both expand to image_token_id, so the tower's features have to arrive
        # in the SAME ORDER as the placeholders in the prompt or the scatter
        # reads shifted features as real ones (silent garbage, the row-148
        # class). Order is therefore taken from the PROMPT below, never from
        # the order these branches happen to run in.
        #
        ids = list(self.tokenizer.encode(text or "", add_special_tokens=False))
        out: Dict[str, Any] = {}
        original_ids = list(ids)
        # (placeholder_position, patches, grids, item spec) per logical visual
        # payload. Merged in prompt order below so the tower rows line up with
        # the scatter even when image and video turns alternate.
        _visual_parts: list = []
        _media_specs: list = []

        def _positions(token_id: int) -> list[int]:
            return [index for index, token in enumerate(original_ids) if token == token_id]

        _first_video_at = (
            ids.index(self.video_token_id) if self.video_token_id in ids else None
        )
        _first_image_at = (
            ids.index(self.image_token_id) if self.image_token_id in ids else None
        )
        if (
            videos is not None
            and images is not None
            and (_first_video_at is None or _first_image_at is None)
        ):
            # Both payloads present but the prompt does not carry both kinds of
            # placeholder: we cannot know where the features belong. Refusing is
            # the only safe answer.
            raise ValueError(
                "dots3_note: video and image payloads were both supplied but the "
                "prompt does not contain both a video and an image placeholder"
            )
        merge = self.image_processor.merge_size
        expected_image_tokens = 0

        if videos is not None:
            frames = _as_frame_sequence(videos)
            if not frames:
                raise ValueError("dots3_note: video input decoded to zero frames")
            patches, grids = self.video_processor(
                frames, fps=kwargs.get("fps"), tokenizer=self.tokenizer
            )
            counts = [merged_token_count(g, merge) for g in grids]
            total = sum(counts)
            # 🚨 The template's <|video_pad|> (151680) becomes IMAGE tokens:
            # frames are independent images, so the expansion emits 151660.
            # config.video_token_id is vestigial at runtime — it must never
            # survive into the ids the model sees.
            ids = expand_media_placeholders(
                ids,
                self.video_token_id,
                [total],
                emit_id=_VIDEO_EXPANSION_SENTINEL,
                modality="video",
            )
            video_spec = {
                "modality": "video",
                "source_index": 0,
                "original_position": int(_first_video_at or 0),
                "token_count": int(total),
                "visual_row_count": int(
                    sum(int(t) * int(h) * int(w) for t, h, w in grids)
                ),
                "visual_grid_count": len(grids),
            }
            _media_specs.append(video_spec)
            _visual_parts.append(
                (
                    video_spec["original_position"],
                    patches,
                    list(grids),
                    video_spec,
                )
            )
            expected_image_tokens += total

        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            patches, grids = self.image_processor(images)
            counts = [merged_token_count(g, merge) for g in grids]
            ids = expand_media_placeholders(
                ids, self.image_token_id, counts, modality="image"
            )
            image_positions = _positions(self.image_token_id)
            row_cursor = 0
            for source_index, (position, grid, token_count) in enumerate(
                zip(image_positions, grids, counts)
            ):
                row_count = int(grid[0]) * int(grid[1]) * int(grid[2])
                image_spec = {
                    "modality": "image",
                    "source_index": source_index,
                    "original_position": int(position),
                    "token_count": int(token_count),
                    "visual_row_count": row_count,
                    "visual_grid_count": 1,
                }
                _media_specs.append(image_spec)
                _visual_parts.append(
                    (
                        int(position),
                        patches[row_cursor : row_cursor + row_count],
                        [grid],
                        image_spec,
                    )
                )
                row_cursor += row_count
            if row_cursor != int(np.asarray(patches).shape[0]):
                raise ValueError(
                    "dots3_note: image grid rows do not reconcile the pixel buffer"
                )
            expected_image_tokens += sum(counts)

        if audio is not None:
            if not isinstance(audio, (list, tuple)):
                audio = [audio]
            waveforms = [self._load_waveform(item) for item in audio]
            feats = self.feature_extractor(waveforms)
            token_lengths = feats["audio_token_lengths"].tolist()
            ids = expand_media_placeholders(
                ids, self.audio_token_id, token_lengths, modality="audio"
            )
            out["input_features"] = feats["input_features"]
            out["chunk_sample_lens"] = feats["chunk_sample_lens"]
            out["chunk_token_lens"] = feats["chunk_token_lens"]
            out["audio_chunk_counts"] = feats["audio_chunk_counts"]
            out["chunk_audio_indices"] = feats["chunk_audio_indices"]
            out["audio_token_lengths"] = feats["audio_token_lengths"]
            audio_positions = _positions(self.audio_token_id)
            chunk_cursor = 0
            chunk_counts = feats["audio_chunk_counts"].tolist()
            for source_index, (position, token_count, chunk_count) in enumerate(
                zip(audio_positions, token_lengths, chunk_counts)
            ):
                audio_spec = {
                    "modality": "audio",
                    "source_index": source_index,
                    "original_position": int(position),
                    "token_count": int(token_count),
                    "audio_chunk_start": int(chunk_cursor),
                    "audio_chunk_end": int(chunk_cursor + int(chunk_count)),
                }
                _media_specs.append(audio_spec)
                chunk_cursor += int(chunk_count)
            if chunk_cursor != int(feats["input_features"].shape[0]):
                raise ValueError(
                    "dots3_note: audio chunk counts do not reconcile input_features"
                )
            placed = sum(1 for t in ids if t == self.audio_token_id)
            if placed != sum(token_lengths):
                raise ValueError(
                    f"dots3_note: {placed} audio token(s) in the prompt but the "
                    f"tower will emit {sum(token_lengths)} embedding row(s)"
                )

        # Resolve each logical item to its exact expanded token interval before
        # replacing the private video sentinel with the public image token.
        # The metadata is consumed by SSD side-keying and partial-hit payload
        # slicing; every field is validated again at the consumer.
        media_items: list = []
        token_cursor = 0
        visual_row_cursor = 0
        visual_grid_cursor = 0
        for spec in sorted(_media_specs, key=lambda item: item["original_position"]):
            modality = spec["modality"]
            emitted_id = (
                _VIDEO_EXPANSION_SENTINEL
                if modality == "video"
                else self.image_token_id
                if modality == "image"
                else self.audio_token_id
            )
            while token_cursor < len(ids) and ids[token_cursor] != emitted_id:
                token_cursor += 1
            token_start = token_cursor
            token_end = token_start + int(spec["token_count"])
            if token_end > len(ids) or any(
                token != emitted_id for token in ids[token_start:token_end]
            ):
                raise ValueError(
                    f"dots3_note: cannot map {modality} payload to expanded tokens"
                )
            token_cursor = token_end
            item = {
                "modality": modality,
                "source_index": int(spec["source_index"]),
                "token_start": int(token_start),
                "token_end": int(token_end),
            }
            if modality in {"image", "video"}:
                row_count = int(spec["visual_row_count"])
                grid_count = int(spec["visual_grid_count"])
                item.update(
                    {
                        "visual_row_start": visual_row_cursor,
                        "visual_row_end": visual_row_cursor + row_count,
                        "visual_grid_start": visual_grid_cursor,
                        "visual_grid_end": visual_grid_cursor + grid_count,
                    }
                )
                visual_row_cursor += row_count
                visual_grid_cursor += grid_count
            else:
                item.update(
                    {
                        "audio_chunk_start": int(spec["audio_chunk_start"]),
                        "audio_chunk_end": int(spec["audio_chunk_end"]),
                    }
                )
            media_items.append(item)

        if _VIDEO_EXPANSION_SENTINEL in ids:
            # Video frames ride the IMAGE token path; the sentinel existed only
            # so the image branch above could tell a real <|imgpad|> apart from
            # an already-expanded video frame.
            ids = [
                self.image_token_id if t == _VIDEO_EXPANSION_SENTINEL else t
                for t in ids
            ]

        if _visual_parts:
            _visual_parts.sort(key=lambda part: part[0])
            _all_patches = [p for _, p, _, _ in _visual_parts]
            _all_grids: list = []
            for _, _, grids_part, _ in _visual_parts:
                _all_grids.extend(grids_part)
            out["pixel_values"] = (
                _all_patches[0]
                if len(_all_patches) == 1
                else np.concatenate([np.asarray(p) for p in _all_patches], axis=0)
            )
            out["image_grid_thw"] = np.asarray(_all_grids, dtype=np.int64)

        if media_items:
            out["_vmlx_dots3_media_items"] = media_items

        if expected_image_tokens:
            placed = sum(1 for t in ids if t == self.image_token_id)
            if placed != expected_image_tokens:
                # Stray literal <|imgpad|> in the prompt or a dropped medium —
                # a mismatched scatter reads shifted features as real ones.
                raise ValueError(
                    f"dots3_note: {placed} image token(s) in the prompt but the "
                    f"tower will emit {expected_image_tokens} embedding row(s)"
                )

        out["input_ids"] = [ids]
        out["attention_mask"] = [[1] * len(ids)]

        if return_tensors == "mlx":
            import mlx.core as mx

            for key in ("input_ids", "attention_mask"):
                out[key] = mx.array(out[key])
            for key in ("pixel_values", "input_features"):
                if key in out:
                    out[key] = mx.array(out[key])
        return out
