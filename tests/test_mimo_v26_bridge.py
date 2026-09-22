"""Small numerical bridge checks; run only when the model GPU lane is idle."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from jang_tools.mimo_v2.v26_model import Model as TextModel, ModelArgs
from vmlx_engine.models.mimo_v26 import Model, MiMoV26Processor


def tiny_text(vocab=256):
    args = ModelArgs(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, v_head_dim=16,
        swa_num_attention_heads=2, swa_num_key_value_heads=1,
        swa_head_dim=32, swa_v_head_dim=16, partial_rotary_factor=0.5,
        hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 1],
        n_routed_experts=4, num_experts_per_tok=2, sliding_window=8,
    )
    mx.random.seed(20260922)
    model = TextModel(args)
    sw = model.layers[1].mlp.switch_mlp
    for name, mode, bits in (("gate_proj", "mxfp4", 4), ("up_proj", "affine", 2), ("down_proj", "affine", 3)):
        setattr(sw, name, getattr(sw, name).to_quantized(group_size=32, bits=bits, mode=mode))
    mx.eval(model.parameters())
    return model


def test_mixed_text_wrapper_cache_and_chunked_prefill():
    text = tiny_text()
    def unexpected():
        pytest.fail("text-only request loaded a tower")
    model = Model(text, SimpleNamespace(vision=unexpected, audio=unexpected), {"model_type": "mimo_v2"})
    ids = mx.array([[1, 4, 7, 11, 9, 6, 2, 8, 10, 15, 14, 13]])
    expected = text(ids)
    actual = model(ids).logits
    assert mx.allclose(expected, actual, atol=1e-5).item()
    cache = model.make_cache()
    pieces = []
    for lo, hi in ((0, 5), (5, 10), (10, 12)):
        pieces.append(model(ids[:, lo:hi], cache=cache).logits)
    assert mx.allclose(expected, mx.concatenate(pieces, axis=1), atol=2e-4, rtol=2e-4).item()


def test_visual_and_audio_features_scatter_to_distinct_slots(monkeypatch):
    from vmlx_engine.models import mimo_v26 as bridge
    # Vocabulary includes the real media ids; hidden width remains small.
    text = tiny_text(vocab=151676)
    visual = mx.full((2, 64), 3.0)
    acoustic = mx.full((1, 64), 7.0)
    omni = SimpleNamespace(
        vision_dtype=mx.float32,
        vision=lambda: (lambda pixels, grid: visual, None),
        audio=lambda: (None, object()),
    )
    monkeypatch.setattr(bridge.audio_port, "embed_audio", lambda codes, encoder: acoustic)
    model = Model(text, omni, {"model_type": "mimo_v2"})
    ids = mx.array([[1, 151656, 2, 151655, 151669, 3]])
    base = text.model.embed_tokens(ids)
    merged = model.get_input_embeddings(
        ids, pixel_values=mx.zeros((8, 1536)), image_grid_thw=[[2, 2, 2]],
        audio_codes=mx.zeros((4, 20), dtype=mx.int32),
    ).inputs_embeds
    assert mx.array_equal(merged[0, [0, 2, 5]], base[0, [0, 2, 5]]).item()
    assert mx.array_equal(merged[0, [1, 3]], visual).item()
    assert mx.array_equal(merged[0, 4], acoustic[0]).item()


def test_native_video_image_order_and_audio_clip_padding(monkeypatch):
    from vmlx_engine.models import mimo_v26 as bridge
    from jang_tools.mimo_v2.v26_vision import MiMoV26VisionProcessor
    ids = [151652, 151656, 151653, 9, 151652, 151655, 151653,
           151673, 151669, 151674, 151673, 151669, 151674]
    tokenizer = SimpleNamespace(chat_template="native", encode=lambda text, **kw: ids if text == "prompt" else [20])
    vp = MiMoV26VisionProcessor()
    omni = SimpleNamespace(vision=lambda: (None, vp), audio=lambda: (object(), None))
    processor = MiMoV26Processor(tokenizer, omni)
    calls = []
    def prepare(wav, sr, tokenizer):
        value = len(calls) + 1
        calls.append(value)
        return mx.full((3, 20), value, dtype=mx.int32), 1
    monkeypatch.setattr(bridge.audio_port, "prepare_audio", prepare)
    out = processor("prompt", images=[np.zeros((64, 64, 3), dtype=np.uint8)],
                    videos=[np.zeros((2, 64, 64, 3), dtype=np.uint8)],
                    video_timestamps=[[0.0, 1.0]],
                    audio=[(np.zeros(240), 24000), (np.zeros(240), 24000)])
    tokens = out["input_ids"][0].tolist()
    assert tokens.index(151656) < tokens.index(151655)
    assert tokens.count(151669) == 2
    assert out["audio_codes"].shape == (8, 20)
    assert mx.all(out["audio_codes"][:4] == 1).item()
    assert mx.all(out["audio_codes"][4:] == 2).item()
