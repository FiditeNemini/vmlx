# SPDX-License-Identifier: Apache-2.0
"""Advertised context capacity must respect what the OS actually has free.

MLX's `active` undercounts what is really WIRED, so `max_ws - active`
overstates the headroom a prompt can use. Measured on MiniMax-M2.7-JANG_K (86GB
weights, 128GB box) at the instant its reconstruct aborted the process: MLX
reported active=82.52GB and 25GB free while the OS showed 97GB wired and free
collapsing to 0-13GB. That ~14.5GB of untracked wiring is why startup
advertised ~69,690 tokens for a model that dies at ~29k and sustains ~20-25k.
"""

import vmlx_engine.server as server


def test_helper_returns_zero_rather_than_guessing(monkeypatch):
    """No psutil must mean 'keep the previous bound', never a made-up number."""
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    assert server._os_available_memory_bytes() == 0


def test_helper_reports_a_plausible_figure():
    value = server._os_available_memory_bytes()
    # Either unavailable (0) or a sane positive byte count.
    assert value == 0 or value > 64 * 1024**2


def test_estimate_takes_the_tighter_of_mlx_and_os(monkeypatch):
    """The MM2.7 case: MLX says 25GB free, the OS says 13GB. The budget must
    be derived from 13GB, not 25GB."""
    import vmlx_engine.utils.memory_limits as ml

    seen_budgets = []

    def _capture(config, budget_bytes, *, max_tokens=0, **kw):
        seen_budgets.append(int(budget_bytes))
        return 1000

    monkeypatch.setattr(ml, "estimate_cache_token_capacity_from_config", _capture)
    monkeypatch.setattr(
        ml, "estimate_dsv4_cache_memory_from_config", lambda cfg, n: None
    )
    monkeypatch.setattr(
        server,
        "_metal_projection_stats",
        lambda: (int(82.52 * 1024**3), int(107.5 * 1024**3)),
    )
    monkeypatch.setattr(
        server, "_loaded_model_config_for_memory_projection", lambda: object()
    )
    monkeypatch.setattr(server, "_declared_context_limit_from_config", lambda cfg: 196608)

    # Generous OS figure -> MLX bound (~25GB) governs.
    monkeypatch.setattr(
        server, "_os_available_memory_bytes", lambda: int(90.0 * 1024**3)
    )
    server._estimate_max_prompt_tokens()
    mlx_bound_budget = seen_budgets[-1]

    # Tight OS figure -> the OS bound governs and the budget must shrink.
    monkeypatch.setattr(
        server, "_os_available_memory_bytes", lambda: int(13.0 * 1024**3)
    )
    server._estimate_max_prompt_tokens()
    os_bound_budget = seen_budgets[-1]

    assert os_bound_budget < mlx_bound_budget, (
        f"OS bound did not tighten the budget: {os_bound_budget} vs {mlx_bound_budget}"
    )
    # 60% of 13GB, the documented KV fraction.
    assert os_bound_budget == int(int(13.0 * 1024**3) * 0.6)


def test_os_bound_is_actually_applied_in_source():
    """Structural pin: the OS figure must clamp `free`, not merely be read."""
    import inspect

    src = inspect.getsource(server._estimate_max_prompt_tokens)
    assert "_os_available_memory_bytes()" in src
    assert "free = min(free, os_available)" in src


def test_a_larger_os_figure_never_loosens_the_mlx_bound(monkeypatch):
    """The OS number may be generous on a quiet box; it must only tighten."""
    import inspect

    src = inspect.getsource(server._estimate_max_prompt_tokens)
    # min() guarantees this, but pin it so a future edit cannot flip to max().
    assert "max(free, os_available)" not in src
