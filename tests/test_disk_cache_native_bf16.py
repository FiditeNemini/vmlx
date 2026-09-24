"""Prompt-cache files must preserve native BF16 bits without FP32 staging."""
import json
import struct

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("strided", [False, True])
def test_prompt_cache_stores_native_bits_and_restores_after_restart(tmp_path, strided):
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.disk_cache import DiskCacheManager

    # Include every BF16 bit pattern, including signed zeros, subnormals,
    # infinities and NaN payloads. Compare integer bits, not float equality.
    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).reshape(1, 1, 8192, 8)
    values = bits.view(mx.bfloat16)
    if strided:
        values = mx.stack([values, values], axis=-1)[..., 0]
    native = KVCache()
    native.keys = values
    native.values = values
    native.offset = 8192
    other = KVCache()
    other.update_and_fetch(mx.ones((1, 1, 8192, 8), dtype=mx.float32),
                           mx.ones((1, 1, 8192, 8), dtype=mx.float16))
    tokens = list(range(8192))
    metadata = {"widened_dtypes": json.dumps({"0.0": "bfloat16"})}
    writer = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=0.1)
    try:
        assert writer.store(tokens, [native, other], metadata=metadata)
        writer._write_queue.join()
        files = list(tmp_path.glob("*.safetensors"))
        assert len(files) == 1
        with files[0].open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for key in ("0.0", "0.1"):
            assert header[key]["dtype"] == "U16"
            begin, end = header[key]["data_offsets"]
            assert end - begin == 65536 * 2
        assert header["1.0"]["dtype"] == "F32"
        assert header["1.1"]["dtype"] == "F16"
    finally:
        writer.shutdown()
    reader = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=0.1)
    try:
        restored = reader.fetch(tokens)
        assert restored is not None
        assert restored[0].offset == native.offset
        for array in (restored[0].keys, restored[0].values):
            assert array.dtype == mx.bfloat16
            assert mx.array_equal(array.view(mx.uint16), bits).item()
        assert restored[1].keys.dtype == mx.float32
        assert restored[1].values.dtype == mx.float16
    finally:
        reader.shutdown()
