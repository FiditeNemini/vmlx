"""Missing scheduler timing is unavailable, never a measured zero (no MLX)."""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize('stats, expected', [({}, None), ({'ewma_ttft_seconds': 9.25}, 9.25), ({'ewma_ttft_seconds': 0.0}, 0.0)])
def test_health_and_cache_preserve_scheduler_timing_availability(stats, expected):
    tree = ast.parse(Path('vmlx_engine/server.py').read_text())
    expressions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'ewma_ttft_seconds':
                    expressions.append(value)
    assert len(expressions) == 2
    for expr in expressions:
        assert eval(compile(ast.Expression(expr), '<production-stats-mapping>', 'eval'),
                    {'stats': stats, 'scheduler_stats': stats}) == expected
