# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


EXTERNAL_MODEL_TEST_FILES = {
    "test_batching_deterministic.py",
    "test_continuous_batching.py",
    "test_emoji_comprehensive.py",
    "test_model_registry.py",
    "test_streaming_detokenizer.py",
}


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vmlx-engine server for integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model loading",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires model loading)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    for item in items:
        if Path(str(item.fspath)).name in EXTERNAL_MODEL_TEST_FILES:
            item.add_marker(pytest.mark.slow)

    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless server URL is explicitly provided
    skip_integration = pytest.mark.skip(reason="Integration tests require --server-url")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def server_url(request):
    """Get server URL from command line."""
    return request.config.getoption("--server-url")


@pytest.fixture(autouse=True)
def _restore_stock_switchglu_after_mimo_fast_path():
    """MiMo-V2 capability discovery (models/mllm.py, model_type mimo_v2 only) installs a process-global SwitchGLU
    decode fast path. A serving process loads one model, but in the suite it leaked into later tests that compute a
    stock SwitchGLU reference (GLM exact-MoE, MiniCPM DSpark verify), which then compared against the wrong arithmetic.
    Undo it after any test that installed it."""
    yield
    try:
        from mlx_lm.models.switch_layers import SwitchGLU
    except Exception:
        return
    original = SwitchGLU.__dict__.get("_vmlx_mimo_original_affine_decode_call")
    if original is not None:
        SwitchGLU.__call__ = original
        del SwitchGLU._vmlx_mimo_original_affine_decode_call
        SwitchGLU._vmlx_mimo_affine_decode_fast_path = False
