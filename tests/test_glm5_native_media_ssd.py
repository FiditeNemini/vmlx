"""Processed-media identity and causal SSD boundaries, without model weights."""
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from tests.test_glm5_companion_disk_codec import native_facade, native_state
from tests.test_glm5_native_ssd_integration import TinyLanguageModel, tiny_processor
from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from vmlx_engine.persistence_outcome import LEDGER
from vmlx_engine.utils.glm5_cache_policy import glm5_native_media_ssd_enabled
from vmlx_engine.utils.glm5_native_media import GLM5_MEDIA_KEY, glm5_media_input_key


TOKENS = [1, 999, 999, 2, 3, 4]


def request(name="native-media", *, video=False):
    req = MLLMBatchRequest(uid=0, request_id=name, prompt="", input_ids=mx.array([TOKENS]))
    setattr(req, "video_pixel_values" if video else "pixel_values", mx.arange(24, dtype=mx.float32).reshape(2, 12))
    setattr(req, "video_grid_thw" if video else "image_grid_thw", mx.array([[1, 2, 4]]))
    req.images = [] if video else ["irrelevant-temp-name"]
    req.videos = ["irrelevant-temp-name"] if video else []
    req._glm_native_full_token_ids = list(TOKENS)
    req._cache_extra_keys = {"request_setting": "preserved"}
    return req


def generator(native):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.native_glm_cache = native
    gen._media_placeholder_token_ids = lambda: {999}
    gen._tokens_contain_media_placeholders = lambda ids: 999 in ids
    return gen


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_identity_is_exact_processed_bytes_not_filename_or_repr(video, dtype):
    req = request(video=video)
    field = "video_pixel_values" if video else "pixel_values"
    value = getattr(req, field).astype(dtype)
    setattr(req, field, value)
    key = glm5_media_input_key(req, TOKENS, {999})
    req.images = ["different-location"] if not video else []
    req.videos = ["different-location"] if video else []
    assert glm5_media_input_key(req, TOKENS, {999}) == key
    changed = value.at[-1, -1].add(1)
    setattr(req, field, changed)
    assert glm5_media_input_key(req, TOKENS, {999}) != key


@pytest.mark.parametrize("change", ["grid", "dtype", "modality", "controls", "ordering"])
def test_grid_dtype_order_modality_and_controls_partition(change):
    req = request()
    key = glm5_media_input_key(req, TOKENS, {999})
    if change == "grid": req.image_grid_thw = mx.array([[1, 4, 2]])
    elif change == "dtype": req.pixel_values = req.pixel_values.astype(mx.bfloat16)
    elif change == "modality": req = request(video=True)
    elif change == "controls": req.image_max_pixels = 100000
    else: req.pixel_values = req.pixel_values[::-1]
    assert glm5_media_input_key(req, TOKENS, {999}) != key


@pytest.mark.parametrize("kind", ["no_pixels", "missing_grid", "bad_grid", "no_tokens", "tail", "audio", "bad_controls"])
def test_unavailable_identity_raises_instead_of_guessing(kind):
    req = request()
    tokens = TOKENS
    if kind == "no_pixels": req.pixel_values = None
    elif kind == "missing_grid": req.image_grid_thw = None
    elif kind == "bad_grid": req.image_grid_thw = mx.array([1, 2])
    elif kind == "no_tokens": tokens = [1, 2, 3]
    elif kind == "tail": tokens = [1, 2, 999]
    elif kind == "audio": req.audio_features = mx.zeros((2, 3))
    else: req.video_fps = float("nan")
    with pytest.raises(ValueError):
        glm5_media_input_key(req, tokens, {999})


def test_default_off_and_unavailable_input_remain_skips(monkeypatch):
    monkeypatch.delenv("VMLX_GLM5_NATIVE_MEDIA_SSD", raising=False)
    assert not glm5_native_media_ssd_enabled()
    gen = generator(Mock())
    req = request("media-default-off")
    gen._prepare_glm_native_media_identity(req, TOKENS)
    assert req._glm_native_media_key is None
    gen._restore_glm_native_prefix(req)
    gen.native_glm_cache.fetch.assert_not_called()
    assert LEDGER.take(req.request_id)["outcome"] == "skipped"
    monkeypatch.setenv("VMLX_GLM5_NATIVE_MEDIA_SSD", "1")
    req.pixel_values = None
    gen._prepare_glm_native_media_identity(req, TOKENS)
    gen._restore_glm_native_prefix(req)
    assert "identity unavailable" in LEDGER.take(req.request_id)["detail"]
    gen.native_glm_cache.fetch.assert_not_called()


@pytest.mark.parametrize("video", [False, True])
def test_only_a_complete_media_prefix_restores_and_drops_payload(tmp_path, monkeypatch, video):
    monkeypatch.setenv("VMLX_GLM5_NATIVE_MEDIA_SSD", "1")
    native = native_facade(tmp_path)
    gen = generator(native)
    req = request(f"media-restored-{video}", video=video)
    gen._prepare_glm_native_media_identity(req, TOKENS)
    try:
        assert req._cache_extra_keys["request_setting"] == "preserved"
        assert req._cache_extra_keys[GLM5_MEDIA_KEY] == req._glm_native_media_key
        gen._store_glm_native_boundary(req, native_state(5))
        assert LEDGER.take(req.request_id)["durable"]
        gen._restore_glm_native_prefix(req)
        assert req._cached_tokens == 5
        assert req.input_ids.tolist() == [[4]]
        assert req.pixel_values is req.video_pixel_values is None
        assert req.image_grid_thw is req.video_grid_thw is None
        assert req._cache_execution["disk_hit"]
        assert native.lookup.total_nbytes == 0
    finally:
        native.close()


def test_hit_before_unconsumed_media_is_not_applied(monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_NATIVE_MEDIA_SSD", "1")
    native = SimpleNamespace(fetch=Mock(return_value=(1, native_state(1))))
    gen = generator(native)
    req = request()
    gen._prepare_glm_native_media_identity(req, TOKENS)
    gen._restore_glm_native_prefix(req)
    assert req.prompt_cache is None
    assert req.input_ids.tolist() == [TOKENS]
    assert req.pixel_values is not None
    assert not getattr(req, "_cached_tokens", 0)


def test_identity_hashes_past_transfer_chunk_boundary():
    req = request()
    req.pixel_values = mx.zeros((2, 140000), mx.float32)
    before = glm5_media_input_key(req, TOKENS, {999})
    req.pixel_values = req.pixel_values.at[-1, -1].add(1)
    assert glm5_media_input_key(req, TOKENS, {999}) != before


def test_changed_pixels_miss_real_disk_while_same_media_text_tail_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_GLM5_NATIVE_MEDIA_SSD", "1")
    native = native_facade(tmp_path)
    gen = generator(native)
    first = request("native-media-first")
    gen._prepare_glm_native_media_identity(first, TOKENS)
    try:
        gen._store_glm_native_boundary(first, native_state(5))
        assert LEDGER.take(first.request_id)["durable"]
        changed = request("native-media-changed")
        changed.pixel_values = changed.pixel_values + 1
        gen._prepare_glm_native_media_identity(changed, TOKENS)
        gen._restore_glm_native_prefix(changed)
        assert changed.prompt_cache is None
        assert native.last_fetch["cached_tokens"] == 0
        same = request("native-media-tail")
        same._glm_native_full_token_ids = TOKENS + [5, 6, 7]
        gen._prepare_glm_native_media_identity(same, same._glm_native_full_token_ids)
        gen._restore_glm_native_prefix(same)
        assert same._cached_tokens == 5
        assert same.input_ids.tolist() == [[4, 5, 6, 7]]
        assert same.pixel_values is None
    finally:
        native.close()


class MediaLM(TinyLanguageModel):
    def __call__(self, inputs, *, cache, inputs_embeds=None, mask=None, return_logits=True):
        return super().__call__(inputs, cache=cache, return_logits=return_logits)


class MediaModel:
    config = TinyLanguageModel.config

    def __init__(self):
        self.language_model = MediaLM()
        self.one_shot = 0

    def get_input_embeddings(self, input_ids, **kwargs):
        return SimpleNamespace(inputs_embeds=mx.zeros((*input_ids.shape, 8)))

    def __call__(self, input_ids, **kwargs):
        self.one_shot += 1
        return self.language_model(input_ids, cache=kwargs["cache"])


@pytest.mark.parametrize("bypass", [False, True])
def test_media_main_pass_splits_at_n_minus_one_even_on_bypass(tmp_path, monkeypatch, bypass):
    monkeypatch.setenv("VMLX_GLM5_NATIVE_MEDIA_SSD", "1")
    native = native_facade(tmp_path)
    model = MediaModel()
    gen = MLLMBatchGenerator(model=model, processor=tiny_processor(), native_glm_cache=native,
                            prefill_step_size=2048, enable_prefix_cache=True, ssm_state_cache_size=0)
    gen._media_placeholder_token_ids = lambda: {999}
    req = request(f"media-main-boundary-{bypass}")
    req._bypass_prefix_cache = bypass
    gen._prepare_glm_native_media_identity(req, TOKENS)
    state = model.language_model.make_cache()
    try:
        output = gen._media_forward(req, req.input_ids, len(TOKENS), state, {"cache": state})
        mx.eval(output)
        assert model.one_shot == 0
        assert model.language_model.calls == [(5, True), (1, True)]
        assert state[1].offset == 6
        if bypass:
            assert native.last_store is None
        else:
            receipt = LEDGER.take(req.request_id)
            assert receipt["durable"] and receipt["retained_tokens"] == 5
            found = native.fetch(TOKENS, extra_keys=req._cache_extra_keys)
            assert found[0] == found[1][1].offset == 5
    finally:
        LEDGER.take(req.request_id)
        native.close()


def test_health_advertises_only_explicit_media_opt_in(tmp_path, monkeypatch):
    from vmlx_engine.server import _native_cache_status
    native = native_facade(tmp_path)
    try:
        monkeypatch.setenv("VMLX_GLM5_NATIVE_MEDIA_SSD", "1")
        status = _native_cache_status(SimpleNamespace(native_glm_cache=native), family="glm5_next")
        assert status["cache_store_policy"]["media"] == "image_video_exact_input_checkpoint"
        assert not status["cache_store_policy"]["media_partial_item_restore"]
    finally:
        native.close()
