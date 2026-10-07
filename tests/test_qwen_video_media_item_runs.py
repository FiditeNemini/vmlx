"""Qwen videos expand to grid_t placeholder runs per source video (audit 2026-10-07, I-15)."""
from types import SimpleNamespace

from vmlx_engine.mllm_batch_generator import _media_placeholder_runs, _qwen_video_media_item_runs

IMG, VID, T = 7, 8, 1  # T: timestamp/text token between video groups
IDS = {"image": {IMG}, "video": {VID}, "audio": set()}


def _tokens():
    # text, image(3), text, video group0(2) ts video group1(2) ts video group2(2), text, image(3)
    return [0, IMG, IMG, IMG, 0, T, VID, VID, T, VID, VID, T, VID, VID, 0, IMG, IMG, IMG, 0]


def test_one_item_per_video_in_prompt_order():
    toks = _tokens()
    runs = _media_placeholder_runs(toks, {IMG, VID})
    assert len(runs) == 5
    req = SimpleNamespace(video_grid_thw=[[3, 2, 2]], extra_kwargs={})
    grouped, sizes = _qwen_video_media_item_runs(req, toks, runs, IDS, model_type="qwen4_exp")
    assert sizes == [1, 3, 1]
    assert grouped == [(1, 4), (6, 14), (15, 18)]


def test_mismatched_temporal_grid_fails_closed():
    toks = _tokens()
    runs = _media_placeholder_runs(toks, {IMG, VID})
    req = SimpleNamespace(video_grid_thw=[[2, 2, 2]], extra_kwargs={})
    assert _qwen_video_media_item_runs(req, toks, runs, IDS, model_type="qwen4_exp") is None
    assert _qwen_video_media_item_runs(req, toks, runs, IDS, model_type="gemma4") is None
