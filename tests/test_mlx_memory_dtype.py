import json
import sys
import types


class _Array:
    def __init__(self, dtype, value):
        self.dtype = dtype
        self.value = value

    def astype(self, dtype):
        return _Array(dtype, self.value)


class _Model:
    def __init__(self, parameters):
        self._parameters = parameters
        self.update_calls = 0

    def parameters(self):
        return self._parameters

    def update(self, parameters):
        self.update_calls += 1
        self._parameters = parameters


class _Mx:
    float16 = "f16"
    bfloat16 = "bf16"
    float32 = "f32"
    uint32 = "u32"
    eval_calls = []

    @classmethod
    def eval(cls, arrays):
        cls.eval_calls.append(arrays)


def _install_fake_mlx_utils(monkeypatch):
    def flatten(tree, prefix=""):
        leaves = []
        if isinstance(tree, dict):
            for key, value in tree.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                leaves.extend(flatten(value, name))
        elif isinstance(tree, list):
            for index, value in enumerate(tree):
                name = f"{prefix}.{index}" if prefix else str(index)
                leaves.extend(flatten(value, name))
        else:
            leaves.append((prefix, tree))
        return leaves

    def unflatten(leaves):
        root = {}
        for name, value in leaves:
            cursor = root
            parts = name.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return root

    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = flatten
    utils.tree_unflatten = unflatten
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)


def _mixed_model():
    return _Model(
        {
            "layer": {
                "q_proj": {
                    "weight": _Array(_Mx.uint32, "packed"),
                    "scales": _Array(_Mx.float16, "q-scales"),
                    "biases": _Array(_Mx.float16, "q-biases"),
                },
                "dense": {
                    "weight": _Array(_Mx.float16, "real-f16-weight"),
                    "biases": _Array(_Mx.float16, "floating-sibling-biases"),
                },
                "norm": {
                    "weight": _Array(_Mx.bfloat16, "norm"),
                    "weight_2": _Array(_Mx.bfloat16, "norm-2"),
                    "weight_3": _Array(_Mx.bfloat16, "norm-3"),
                    "weight_4": _Array(_Mx.bfloat16, "norm-4"),
                },
            },
            "mtp": {"weight": _Array(_Mx.float16, "mtp-real-f16")},
        }
    )


def _f16_compute_model():
    """Step-style artifact: BF16 config, but every real compute anchor is F16."""
    return _Model(
        {
            "layer": {
                "q_proj": {
                    "weight": _Array(_Mx.uint32, "packed"),
                    "scales": _Array(_Mx.float16, "q-scales"),
                    "biases": _Array(_Mx.float16, "q-biases"),
                },
                "input_layernorm": {
                    "weight": _Array(_Mx.float16, "f16-norm"),
                },
                "q_norm": {
                    "weight": _Array(_Mx.float16, "f16-q-norm"),
                },
            },
        }
    )


def test_harmonize_casts_only_packed_quant_metadata(monkeypatch):
    from vmlx_engine.mlx_memory import harmonize_quant_metadata_dtypes

    _install_fake_mlx_utils(monkeypatch)
    _Mx.eval_calls = []
    model = _mixed_model()

    result = harmonize_quant_metadata_dtypes(
        model,
        declared_dtype="bfloat16",
        mx=_Mx,
    )

    params = model.parameters()
    assert result == {
        "f16": 5,
        "bf16": 4,
        "f32": 0,
        "eligible": 2,
        "cast": 2,
        "preserved_f16": 3,
        "anchor_f16": 3,
        "anchor_bf16": 4,
        "anchor_f32": 0,
        "anchor_policy_match": 1,
    }
    assert params["layer"]["q_proj"]["scales"].dtype == _Mx.bfloat16
    assert params["layer"]["q_proj"]["biases"].dtype == _Mx.bfloat16
    assert params["layer"]["dense"]["biases"].dtype == _Mx.float16
    assert params["layer"]["dense"]["weight"].dtype == _Mx.float16
    assert params["mtp"]["weight"].dtype == _Mx.float16
    assert model.update_calls == 1
    assert len(_Mx.eval_calls) == 1


def test_harmonize_does_not_create_bf16_f16_mixture_without_bf16_anchors(monkeypatch):
    from vmlx_engine.mlx_memory import harmonize_quant_metadata_dtypes

    _install_fake_mlx_utils(monkeypatch)
    _Mx.eval_calls = []
    model = _f16_compute_model()

    result = harmonize_quant_metadata_dtypes(
        model,
        declared_dtype="bfloat16",
        mx=_Mx,
    )

    assert result["eligible"] == 2
    assert result["bf16"] == 0
    assert result["cast"] == 0
    assert result["preserved_f16"] == 4
    assert model.parameters()["layer"]["q_proj"]["scales"].dtype == _Mx.float16
    assert model.parameters()["layer"]["q_proj"]["biases"].dtype == _Mx.float16
    assert model.update_calls == 0
    assert _Mx.eval_calls == []


def test_harmonize_requires_bundle_declared_bfloat16(monkeypatch, tmp_path):
    from vmlx_engine.mlx_memory import harmonize_quant_metadata_dtypes

    _install_fake_mlx_utils(monkeypatch)
    (tmp_path / "config.json").write_text(
        json.dumps({"dtype": "float16"}),
        encoding="utf-8",
    )
    model = _mixed_model()

    result = harmonize_quant_metadata_dtypes(
        model,
        model_path=tmp_path,
        mx=_Mx,
    )

    assert result["cast"] == 0
    assert result["preserved_f16"] == 5
    assert model.update_calls == 0


def test_wrapper_is_on_by_default_for_declared_bfloat16(monkeypatch, tmp_path):
    from vmlx_engine.mlx_memory import maybe_harmonize_quant_metadata_dtypes

    monkeypatch.delenv("VMLX_HARMONIZE_PARAM_DTYPES", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"dtype": "bfloat16"}}),
        encoding="utf-8",
    )
    _install_fake_mlx_utils(monkeypatch)
    model = _mixed_model()

    result = maybe_harmonize_quant_metadata_dtypes(
        model,
        model_path=tmp_path,
        mx=_Mx,
    )

    assert result is not None
    assert result["cast"] == 2
    assert model.update_calls == 1
    assert model._vmlx_quant_metadata_dtype_harmonization == {
        "enabled": True,
        "policy": "bundle_declared_bfloat16_with_runtime_bfloat16_anchors",
        "explicit": False,
        **result,
    }


def test_wrapper_can_be_explicitly_disabled(monkeypatch):
    from vmlx_engine.mlx_memory import maybe_harmonize_quant_metadata_dtypes

    monkeypatch.setenv("VMLX_HARMONIZE_PARAM_DTYPES", "off")
    model = _mixed_model()

    assert maybe_harmonize_quant_metadata_dtypes(model, mx=_Mx) is None
    assert model.update_calls == 0
    assert model._vmlx_quant_metadata_dtype_harmonization == {
        "enabled": False,
        "reason": "explicitly_disabled",
    }


def test_dtype_harmonization_runs_at_both_loader_boundaries_not_server_startup():
    from pathlib import Path

    root = Path(__file__).parents[1]
    tokenizer_source = (root / "vmlx_engine/utils/tokenizer.py").read_text()
    mllm_source = (root / "vmlx_engine/models/mllm.py").read_text()
    server_source = (root / "vmlx_engine/server.py").read_text()

    assert "maybe_harmonize_quant_metadata_dtypes" in tokenizer_source
    assert "maybe_harmonize_quant_metadata_dtypes" in mllm_source
    assert "maybe_harmonize_quant_metadata_dtypes" not in server_source
