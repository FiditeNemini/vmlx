"""CPU-only cache-identity and dispatch wiring; no MLX import or numeric claim."""

import ast
import builtins
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
FLAGS = {
    "QWEN4_PREFILL_FUSED": "1",
    "WEIGHTED_UNSORT": "0",
    "TAIL_SPLIT": "1",
}


def load_identity():
    path = ROOT / "vmlx_engine/jangh/runtime_identity.py"
    spec = importlib.util.spec_from_file_location("isolated_qwen_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,default", FLAGS.items())
@pytest.mark.parametrize("value", [None, "", "0", "1", "true", "false", " 1 ", "2"])
def test_exact_producer_flag_semantics_and_identity(monkeypatch, name, default, value):
    env = "JANGH_" + name
    monkeypatch.delenv(env, raising=False)
    if value is not None:
        monkeypatch.setenv(env, value)
    policy = load_identity()
    expected = "1" if (default if value is None else value) == "1" else "0"
    assert getattr(policy, name) == expected
    identity = policy.runtime_identity()
    assert f";{name.lower()}={expected}" in identity
    monkeypatch.setenv(env, "0" if expected == "1" else "1")
    assert policy.runtime_identity() == identity
    assert load_identity().runtime_identity() != identity
    monkeypatch.setenv(env, expected)
    assert load_identity().runtime_identity() == identity


def execution_namespace(policy):
    def import_owner(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("runtime_identity"):
            return policy
        if name == "vmlx_engine.jangh.contract":
            return SimpleNamespace(validate_format=lambda config: True)
        if name == "vmlx_engine.jangh.install":
            return SimpleNamespace(install_jangh=lambda model, config: 1)
        return builtins.__import__(name, globals, locals, fromlist, level)

    return {
        "__builtins__": dict(vars(builtins), __import__=import_owner),
        "os": os,
        "logger": logging.getLogger(__name__),
    }


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_loader_uses_frozen_policy_after_environment_mutation(monkeypatch, enabled):
    monkeypatch.setenv("JANGH_QWEN4_PREFILL_FUSED", enabled)
    policy = load_identity()
    monkeypatch.setenv("JANGH_QWEN4_PREFILL_FUSED", "1" if enabled == "0" else "0")
    source = ROOT / "vmlx_engine/models/qwen4_exp/loader.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_install_jangh_routed_experts")
    namespace = execution_namespace(policy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    bank = SimpleNamespace(is_jangtq2=True, gate_proj=SimpleNamespace(),
                           up_proj=SimpleNamespace(), down_proj=SimpleNamespace())
    model = SimpleNamespace(named_modules=lambda: [("bank", bank)])
    assert namespace[node.name](model, {}) == 1
    assert getattr(bank, "use_weighted_unsort", False) == (enabled == "1")
    for linear in (bank.gate_proj, bank.up_proj, bank.down_proj):
        assert getattr(linear, "use_h32_rows", False) == (enabled == "1")


@pytest.mark.parametrize("filename,name", [("switch.py", "WEIGHTED_UNSORT"),
                                          ("kernels.py", "TAIL_SPLIT")])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_dispatch_imports_same_frozen_policy(monkeypatch, filename, name, enabled):
    monkeypatch.setenv("JANGH_" + name, enabled)
    policy = load_identity()
    monkeypatch.setenv("JANGH_" + name, "1" if enabled == "0" else "0")
    source = ROOT / "vmlx_engine/jangh" / filename
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module == "runtime_identity":
            if any(alias.name == name for alias in node.names):
                nodes.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            nodes.append(node)
    assert nodes, "Dispatch must consume the persisted-state policy"
    namespace = execution_namespace(policy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    expected = enabled == "1" if name == "TAIL_SPLIT" else enabled
    assert namespace[name] == expected
