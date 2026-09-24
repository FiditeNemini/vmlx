from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from vmlx_engine.models.mimo_v26_cache import media_item_runs,prepare_media_tail

IDS={'image':{11},'video':{12},'audio':{13}}

def fixture():
    tokens=[1,11,11,2,3,12,12,4,12,12,5,13,13,6,7]
    records=[
        dict(modality='image',source_index=0,token_start=1,token_end=3,pad_tokens=2,pixel_start=0,pixel_end=8),
        dict(modality='video',source_index=0,token_start=5,token_end=10,pad_tokens=4,pixel_start=8,pixel_end=24),
        dict(modality='audio',source_index=0,token_start=11,token_end=13,pad_tokens=2,code_start=0,code_end=8)]
    req=SimpleNamespace(images=['image-bytes'],videos=['video-bytes'],audio=['audio-bytes'],
        extra_kwargs={'_vmlx_mimo26_media_items':records},pixel_values=np.arange(24*3).reshape(24,3),
        image_grid_thw=np.array([[1,2,4],[2,2,4]]),audio_codes=np.arange(8*20).reshape(8,20))
    return req,tokens

def test_prompt_order_groups_timestamped_video_and_keeps_audio_identity():
    req,tokens=fixture();ranges,sizes,assignments=media_item_runs(req,tokens,IDS)
    assert ranges==[(1,3),(5,10),(11,13)]
    assert sizes==[1,2,1]
    assert assignments==[('image','image-bytes'),('video','video-bytes'),('audio','audio-bytes')]

def test_tail_after_image_keeps_whole_video_and_audio():
    req,tokens=fixture();pixels=req.pixel_values.copy();codes=req.audio_codes.copy()
    result=prepare_media_tail(req,tokens,4,IDS)
    assert result['covered_media_items']==1
    np.testing.assert_array_equal(req.pixel_values,pixels[8:])
    np.testing.assert_array_equal(req.image_grid_thw,[[2,2,4]])
    np.testing.assert_array_equal(req.audio_codes,codes)

def test_mid_video_or_bad_patch_geometry_declines_without_mutation():
    for boundary,damage in [(8,False),(4,True)]:
        req,tokens=fixture();old=req.pixel_values;old_codes=req.audio_codes
        if damage:req.extra_kwargs['_vmlx_mimo26_media_items'][1]['pixel_end']=23
        assert prepare_media_tail(req,tokens,boundary,IDS) is None
        assert req.pixel_values is old and req.audio_codes is old_codes

def test_tail_after_video_releases_visual_payload_and_keeps_audio():
    req,tokens=fixture();result=prepare_media_tail(req,tokens,10,IDS)
    assert result['covered_media_items']==2
    assert req.pixel_values is None and req.image_grid_thw is None
    assert req.audio_codes.shape==(8,20)

def test_missing_or_duplicate_source_and_wrong_tokens_fail_closed():
    req,tokens=fixture();req.images=[]
    assert media_item_runs(req,tokens,IDS) is None
    req,tokens=fixture();tokens[6]=13
    assert media_item_runs(req,tokens,IDS) is None
