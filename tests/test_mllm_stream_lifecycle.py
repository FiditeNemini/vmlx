"""MLLM generation stream lifecycle contracts."""

from __future__ import annotations

import inspect


def test_reset_generation_streams_clears_module_and_class_handles():
    """Deep sleep/reload must not keep thread-local MLX streams from old workers."""
    import vmlx_engine.mllm_batch_generator as gen

    old_module_stream = gen._GENERATION_STREAM
    old_module_owner = gen._GENERATION_STREAM_OWNER
    old_class_stream = gen.MLLMBatchGenerator._stream
    try:
        gen._GENERATION_STREAM = object()
        gen._GENERATION_STREAM_OWNER = 123
        gen.MLLMBatchGenerator._stream = object()

        gen.reset_generation_streams()

        assert gen._GENERATION_STREAM is None
        assert gen._GENERATION_STREAM_OWNER is None
        assert gen.MLLMBatchGenerator._stream is None
    finally:
        gen._GENERATION_STREAM = old_module_stream
        gen._GENERATION_STREAM_OWNER = old_module_owner
        gen.MLLMBatchGenerator._stream = old_class_stream


def test_reset_generation_streams_clears_direct_vlm_handle():
    """Direct mlx-vlm streams are thread-local and must reset with MLLM streams."""
    import vmlx_engine.models.mllm as mllm
    import vmlx_engine.server as srv

    old_stream = mllm._VLM_STREAM
    old_owner = mllm._VLM_STREAM_OWNER
    try:
        mllm._VLM_STREAM = object()
        mllm._VLM_STREAM_OWNER = 123

        srv._reset_mllm_generation_streams()

        assert mllm._VLM_STREAM is None
        assert mllm._VLM_STREAM_OWNER is None
    finally:
        mllm._VLM_STREAM = old_stream
        mllm._VLM_STREAM_OWNER = old_owner


def test_direct_vlm_stream_rebinds_mlx_vlm_global_and_wraps_nonstream_generate():
    """Direct non-stream reloads must not synchronize an old worker's stream."""
    import vmlx_engine.models.mllm as mllm

    vlm_stream_source = inspect.getsource(mllm._vlm_stream)
    generate_source = inspect.getsource(mllm.MLXMultimodalLM.generate)

    assert 'importlib.import_module("mlx_vlm.generate")' in vlm_stream_source
    assert "_VLM_STREAM_OWNER != owner" in vlm_stream_source
    assert "_mvg.generation_stream = stream" in vlm_stream_source
    assert "with _MaybeVLMStream():" in generate_source
    assert generate_source.index("with _MaybeVLMStream():") < generate_source.index(
        "result = generate("
    )


def test_all_mllm_stream_rebinds_target_the_generate_submodule():
    """Package-level ``generate`` is a function and must never receive the stream."""
    import vmlx_engine.engine.batched as batched
    import vmlx_engine.mllm_batch_generator as batch_generator

    batch_source = inspect.getsource(batch_generator._gen_stream)
    fallback_source = inspect.getsource(batched.BatchedEngine._simple_mllm_chat_output)

    for source in (batch_source, fallback_source):
        assert 'importlib.import_module("mlx_vlm.generate")' in source
        assert "import mlx_vlm.generate as" not in source
    assert "_GENERATION_STREAM_OWNER != owner" in batch_source


def test_mlx_vlm_compat_stream_patch_keeps_live_module_globals():
    """A main-thread compat patch must not freeze mlx-vlm's stream global."""
    import importlib

    from vmlx_engine.utils import mlx_vlm_compat

    generate_mod = importlib.import_module("mlx_vlm.generate")
    mlx_vlm_compat.apply()

    patched = generate_mod.stream_generate
    assert getattr(patched, "_vmlx_rank3_cache_trim_patched", False)
    assert patched.__globals__ is generate_mod.__dict__

    old_stream = generate_mod.generation_stream
    replacement = object()
    try:
        generate_mod.generation_stream = replacement
        assert patched.__globals__["generation_stream"] is replacement
    finally:
        generate_mod.generation_stream = old_stream


def test_simple_engine_establishes_owned_vlm_stream_before_model_load():
    """Lazy MLLM load graphs must not capture a replacement worker's Stream 0."""
    import vmlx_engine.engine.simple as simple

    start_source = inspect.getsource(simple.SimpleEngine.start)
    assert "with _MaybeVLMStream():" in start_source
    assert start_source.index("with _MaybeVLMStream():") < start_source.index(
        "model.load()"
    )


def test_server_resets_mllm_streams_when_replacing_or_unloading_engine():
    """Model switch and deep sleep both tear down MLLM stream ownership."""
    import vmlx_engine.server as srv

    load_body = inspect.getsource(srv.load_model)
    deep_sleep_body = inspect.getsource(srv.admin_deep_sleep)

    assert "_reset_mllm_generation_streams()" in load_body
    assert "_reset_mllm_generation_streams()" in deep_sleep_body
