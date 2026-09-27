"""A live batch advancing must not mutate a saved companion's padding."""
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import ArraysCache

from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache


@pytest.mark.parametrize("mutation_owner", ["stored_source", "fetched_copy"])
def test_native_advance_preserves_companion_padding(mutation_owner):
    companion = SSMCompanionCache(max_entries=2, disk_store=False)
    source = ArraysCache(2, left_padding=[3])
    source.cache[0] = mx.arange(4, dtype=mx.float32).reshape(1, 4)
    source.prepare(lengths=[8])
    companion.store([1, 2], 2, [source])

    if mutation_owner == "stored_source":
        mutable = source
    else:
        states, complete = companion.fetch([1, 2], 2)
        assert complete
        mutable = states[0]
    mutable.advance(1)
    mx.eval(mutable.left_padding, mutable.lengths)
    assert mutable.left_padding.tolist() == [2]
    assert mutable.lengths.tolist() == [7]

    states, complete = companion.fetch([1, 2], 2)
    assert complete
    assert states[0].left_padding.tolist() == [3]
    assert states[0].lengths.tolist() == [8]
    assert states[0].cache[0].tolist() == [[0, 1, 2, 3]]
    assert states[0].cache[1] is None
