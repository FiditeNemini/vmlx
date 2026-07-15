import copy

import numpy as np

from vmlx_engine.utils.jang_affine_storage import (
    affine1_storage_modules,
    expand_affine1_shard_mlx,
    prepare_affine1_runtime_config,
)
from vmlx_engine.utils.jang_loader import _post_load_quantization_overrides


def _metadata():
    module = "language_model.model.layers.0.self_attn.q_proj"
    config = {
        "quantization": {
            "bits": 1,
            "group_size": 128,
            module: {"bits": 1, "group_size": 128, "mode": "affine"},
            "vision_tower.blocks.0.attn.qkv": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        }
    }
    jang_config = {
        "quantization": {
            "method": "jang-affine-discrete",
            "tensor_quantization_manifest_schema": 2,
            "tensor_quantization_manifest": {
                module: {"bits": 1, "storage_bits": 1, "group_size": 128},
                "vision_tower.blocks.0.attn.qkv": {
                    "bits": 4,
                    "storage_bits": 4,
                    "group_size": 64,
                },
            }
        }
    }
    return module, config, jang_config


def test_affine1_runtime_config_is_in_memory_and_module_scoped():
    module, config, jang_config = _metadata()
    original = copy.deepcopy(config)

    runtime, modules = prepare_affine1_runtime_config(config, jang_config)

    assert modules == frozenset({module})
    assert affine1_storage_modules(jang_config) == modules
    assert runtime["quantization"]["bits"] == 2
    assert runtime["quantization"][module]["bits"] == 2
    assert runtime["quantization"][module]["storage_bits"] == 1
    assert runtime["quantization"]["vision_tower.blocks.0.attn.qkv"]["bits"] == 4
    assert config == original


def test_affine1_shard_expansion_preserves_codes_and_sidecars():
    import mlx.core as mx

    module, _, _ = _metadata()
    source = np.array([[0x01234567, 0x89ABCDEF]], dtype=np.uint32)
    scales = mx.array([[0.25]], dtype=mx.float16)
    biases = mx.array([[-0.125]], dtype=mx.float16)
    weights = {
        f"{module}.weight": mx.array(source),
        f"{module}.scales": scales,
        f"{module}.biases": biases,
    }

    expanded, count = expand_affine1_shard_mlx(weights, {module})
    mx.eval(expanded[f"{module}.weight"])

    result = np.array(expanded[f"{module}.weight"])
    expected_codes = [int((source[0, 0] >> bit) & 1) for bit in range(32)]
    expected_codes += [int((source[0, 1] >> bit) & 1) for bit in range(32)]
    actual_codes = [
        int((result[0, word // 16] >> (2 * (word % 16))) & 0b11)
        for word in range(64)
    ]
    assert count == 1
    assert result.shape == (1, 4)
    assert actual_codes == expected_codes
    assert expanded[f"{module}.scales"] is scales
    assert expanded[f"{module}.biases"] is biases


def test_discrete_affine_manifest_drives_post_load_mixed_precision():
    module, config, jang_config = _metadata()

    overrides = _post_load_quantization_overrides(config, jang_config)

    assert overrides[module] == {"bits": 2, "group_size": 128}
    assert overrides["vision_tower.blocks.0.attn.qkv"] == {
        "bits": 4,
        "group_size": 64,
    }
