"""Numerical startup modes must not restore each other's persisted cache."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
import json, os, sys
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from vmlx_engine import prefix_cache
from vmlx_engine.disk_cache import DiskCacheManager
from vmlx_engine.qwen4_rope_policy import (
    exact_rope_attn_enabled, exact_rope_attn_status,
)
mode = exact_rope_attn_enabled()
identity = prefix_cache.runtime_cache_fingerprint()
os.environ['VMLX_QWEN4_EXACT_ROPE_ATTN'] = '0' if mode else '1'
assert exact_rope_attn_enabled() == mode
assert exact_rope_attn_status()['enabled'] == mode
assert prefix_cache.runtime_cache_fingerprint() == identity
manager = DiskCacheManager(sys.argv[1], max_size_gb=0.01)
tokens = list(range(8))
try:
    if sys.argv[2] == 'write':
        cache = KVCache()
        values = mx.arange(32, dtype=mx.float32).reshape(1, 1, 8, 4)
        cache.update_and_fetch(values, values + 1)
        mx.eval(cache.state)
        assert manager.store(tokens, [cache])
        assert manager.flush_pending_writes(tokens)
    restored = manager.fetch(tokens)
    if restored is not None:
        assert restored[0].offset == 8
        assert restored[0].keys[0, 0, :8, :].tolist() == [list(range(i, i+4)) for i in range(0,32,4)]
    print(json.dumps({'mode':mode, 'identity':identity, 'hit':restored is not None}))
finally:
    manager.shutdown()
'''


def probe(cache_dir, mode, action):
    env = {**os.environ, 'VMLX_QWEN4_EXACT_ROPE_ATTN': str(mode)}
    result = subprocess.run(
        [sys.executable, '-c', PROBE, str(cache_dir), action],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize('written_mode', [0, 1])
def test_restart_restores_only_same_rotary_mode(tmp_path, written_mode):
    pytest.importorskip('mlx.core')
    written = probe(tmp_path, written_mode, 'write')
    restarted = probe(tmp_path, written_mode, 'read')
    opposite = probe(tmp_path, 1-written_mode, 'read')
    assert written['hit'] and restarted['hit']
    assert written['identity'] == restarted['identity']
    assert opposite['identity'] != written['identity']
    assert not opposite['hit']


@pytest.mark.parametrize('mode', [0, 1])
def test_health_contract_reports_effective_rotary_mode(tmp_path, mode):
    pytest.importorskip('mlx.core')
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen4_exp'}))
    code = r"""
import json, os, sys
from vmlx_engine.qwen4_rope_policy import exact_rope_attn_enabled
expected = exact_rope_attn_enabled()
os.environ['VMLX_QWEN4_EXACT_ROPE_ATTN'] = '0' if expected else '1'
from vmlx_engine.server import _family_acceleration_contract
status = _family_acceleration_contract(sys.argv[1])['numerical_policy']['attention_exact_rope']
assert status['enabled'] == expected
assert status['configuration_scope'] == 'process_startup'
print(json.dumps(status))
"""
    result = subprocess.run(
        [sys.executable, '-c', code, str(tmp_path)], cwd=ROOT,
        env={**os.environ, 'VMLX_QWEN4_EXACT_ROPE_ATTN': str(mode)},
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['enabled'] == bool(mode)


def test_vendored_attention_alias_resolves_engine_policy():
    import ast
    from vmlx_engine.qwen4_rope_policy import exact_rope_attn_enabled

    source = ROOT / 'vmlx_engine/models/qwen4_exp/language.py'
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == '_qsa_exact_rope_attn_enabled')
    # The registry also loads this exact source under the mlx_vlm alias.
    namespace = {'__package__': 'mlx_vlm.models.qwen4_exp'}
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(source), 'exec'), namespace)
    assert namespace[owner.name]() == exact_rope_attn_enabled()
