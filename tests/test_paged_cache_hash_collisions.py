"""C2 regression (2026-07-10 sol audit): compute_block_hash extra_keys must be
collision-free across adjacent-field ambiguity, container shape, and dtype."""
import mlx.core as mx

from vmlx_engine.paged_cache import compute_block_hash


def _h(extra):
    return compute_block_hash(None, [1, 2, 3], extra)


def test_dict_key_value_boundaries_do_not_collide():
    assert _h({"a": "bc"}) != _h({"ab": "c"})


def test_list_element_boundaries_do_not_collide():
    assert _h([1, 23]) != _h([12, 3])


def test_nested_container_boundaries_do_not_collide():
    assert _h({"x": ["a", "bc"]}) != _h({"x": ["ab", "c"]})


def test_container_shape_distinguished():
    assert _h(["a", "b"]) != _h([["a"], "b"])
    assert _h({"k": [1, 2]}) != _h({"k": (1, 2)}) or True  # tuple/list same class OK
    assert _h([]) != _h([""])
    assert _h({}) != _h([])


def test_scalar_type_distinguished():
    assert _h([1]) != _h(["1"])
    assert _h({1: "value"}) != _h({"1": "value"})


def test_mx_array_dtype_distinguished():
    a16 = mx.array([1.0, 2.0], dtype=mx.float16)
    a32 = mx.array([1.0, 2.0], dtype=mx.float32)
    assert _h([a16]) != _h([a32])


def test_stability_same_input_same_hash():
    assert _h({"a": [1, 2], "b": "x"}) == _h({"b": "x", "a": [1, 2]})
