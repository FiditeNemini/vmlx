import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_gate_module():
    path = Path("panel/scripts/release-gate-python-app.py").resolve()
    spec = importlib.util.spec_from_file_location("release_gate_python_app", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_release_gate_loop_detector_catches_word_loop():
    gate = _load_gate_module()
    assert gate.obvious_loop("state " * 80)


def test_release_gate_loop_detector_catches_no_space_cjk_phrase_loop():
    gate = _load_gate_module()
    text = "音苷苷和音诺族的对策" * 80
    assert gate.obvious_loop(text)


def test_release_gate_loop_detector_catches_emoji_loop():
    gate = _load_gate_module()
    assert gate.obvious_loop("👀" * 200)


def test_release_gate_loop_detector_allows_short_clean_answer():
    gate = _load_gate_module()
    assert not gate.obvious_loop("Paris is the capital of France.")


class _FakeGate:
    def __init__(self, stdout: str):
        self.stdout = stdout
        self.records = []
        self.run_cmd = None
        self.log_dir = Path("/isolated-release-gate")
        self.run_kwargs = None

    def run(self, name, cmd, **kwargs):
        self.run_cmd = cmd
        self.run_kwargs = kwargs
        self.records.append((name, "RUN", kwargs))
        return subprocess.CompletedProcess(cmd, 0, self.stdout, "")

    def record(self, name, status, detail=""):
        self.records.append((name, status, detail))


class _SequenceGate:
    def __init__(self, stdout_by_name: dict[str, str]):
        self.stdout_by_name = stdout_by_name
        self.records = []
        self.run_calls = []
        self.log_dir = Path("/isolated-release-gate")

    def run(self, name, cmd, **kwargs):
        self.run_calls.append((name, cmd, kwargs))
        self.records.append((name, "RUN", kwargs))
        return subprocess.CompletedProcess(cmd, 0, self.stdout_by_name.get(name, ""), "")

    def record(self, name, status, detail=""):
        self.records.append((name, status, detail))


def _developer_id_signature_stdout() -> str:
    return "\n".join(
        [
            "Executable=/tmp/vMLX.app/Contents/MacOS/vMLX",
            "CodeDirectory v=20500 size=325 flags=0x10000(runtime) hashes=4+3 location=embedded",
            "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)",
            "Authority=Developer ID Certification Authority",
            "Authority=Apple Root CA",
            "TeamIdentifier=55KGF2S5AY",
        ]
    )


def _release_entitlements_stdout(*, missing: str | None = None) -> str:
    keys = [
        "com.apple.security.cs.allow-jit",
        "com.apple.security.cs.allow-unsigned-executable-memory",
        "com.apple.security.cs.disable-library-validation",
        "com.apple.security.network.client",
        "com.apple.security.files.user-selected.read-write",
    ]
    keys = [key for key in keys if key != missing]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<plist version="1.0">',
        "<dict>",
    ]
    for key in keys:
        lines.extend([f"<key>{key}</key>", "<true/>"])
    lines.extend(["</dict>", "</plist>"])
    return "\n".join(lines)


def test_packaged_bundled_version_parity_passes_when_import_version_matches():
    gate_module = _load_gate_module()
    gate = _FakeGate("import ok\n1.5.25\n")

    gate_module.check_packaged_bundled_import_version(
        gate, Path("/app/python3"), "1.5.25", "1.5.25"
    )

    assert gate.records[-1] == (
        "packaged bundled version",
        "PASS",
        "app=1.5.25, bundled=1.5.25, expected=1.5.25",
    )
    assert "mflux" in " ".join(gate.run_cmd)
    assert gate.run_kwargs["cwd"] == gate.log_dir
    assert gate.run_kwargs["env"]["PYTHONPATH"] == ""
    assert gate.run_kwargs["env"]["PYTHONNOUSERSITE"] == "1"


def test_packaged_bundled_engine_path_uses_isolated_cwd():
    gate_module = _load_gate_module()
    gate = _FakeGate("/app/Contents/Resources/bundled-python/python/lib/python3.12/site-packages/vmlx_engine\n")

    path = gate_module.packaged_engine_dir(gate, Path("/app/python3"))

    assert path == Path("/app/Contents/Resources/bundled-python/python/lib/python3.12/site-packages/vmlx_engine")
    assert gate.run_kwargs["cwd"] == gate.log_dir
    assert gate.run_kwargs["env"]["PYTHONPATH"] == ""


def test_release_gate_jang_source_prefers_documented_env(monkeypatch):
    gate_module = _load_gate_module()
    monkeypatch.setenv("VMLX_JANG_TOOLS_SOURCE", "/clean/documented")
    monkeypatch.setenv("VMLINUX_JANG_TOOLS_SOURCE", "/legacy")

    assert gate_module.jang_tools_source_root() == Path("/clean/documented")


def test_release_gate_jang_source_keeps_legacy_env_fallback(monkeypatch):
    gate_module = _load_gate_module()
    monkeypatch.delenv("VMLX_JANG_TOOLS_SOURCE", raising=False)
    monkeypatch.setenv("VMLINUX_JANG_TOOLS_SOURCE", "/legacy")

    assert gate_module.jang_tools_source_root() == Path("/legacy")


def test_release_gate_objective_digest_default_tracks_current_release_matrix():
    gate_module = _load_gate_module()
    gate = _FakeGate('{"requirements":[]}\n')

    gate_module.check_objective_proof_digest(gate)

    assert gate.run_cmd[-2:] == [
        "--out",
        str(
            Path.cwd()
            / "build/current-objective-proof-after-pr-intake-matrix-refresh-20260609.json"
        ),
    ]


def test_release_gate_twine_command_skips_installed_app_console_script(
    monkeypatch, tmp_path
):
    gate_module = _load_gate_module()
    stale_twine = tmp_path / "twine"
    stale_twine.write_text(
        "#!/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3\n",
        encoding="utf-8",
    )
    stale_twine.chmod(0o755)
    fallback_python = tmp_path / "python"
    fallback_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fallback_python.chmod(0o755)

    monkeypatch.delenv("TWINE", raising=False)
    monkeypatch.setattr(gate_module, "python_has_module", lambda *_: False)
    monkeypatch.setattr(gate_module.shutil, "which", lambda name: str(stale_twine))
    monkeypatch.setattr(gate_module.sys, "executable", str(fallback_python))

    assert gate_module.twine_command() == [str(fallback_python), "-m", "twine"]


def test_release_gate_twine_env_disables_user_site_packages():
    gate_module = _load_gate_module()
    gate = _FakeGate("")

    env = gate_module.twine_env(gate)

    assert env["PYTHONPATH"] == ""
    assert env["PYTHONNOUSERSITE"] == "1"


def test_live_engine_gate_uses_packaged_python_with_isolated_cwd():
    src = Path("panel/scripts/release-gate-python-app.py").read_text()
    assert 'cwd=str(gate.log_dir)' in src
    assert '"PYTHONPATH": ""' in src
    assert '"PYTHONNOUSERSITE": "1"' in src
    assert '"-m",' in src and '"vmlx_engine.cli"' in src
    assert '"--continuous-batching"' in src


def test_live_engine_gate_uses_artifact_local_cache_dirs():
    """Live gates must not pass because a default model cache was warm already."""
    src = Path("panel/scripts/release-gate-python-app.py").read_text()

    assert '"--disk-cache-dir"' in src
    assert 'str(gate.log_dir / "prompt-cache")' in src
    assert '"--block-disk-cache-dir"' in src
    assert 'str(gate.log_dir / "block-cache")' in src


def test_live_engine_gate_requires_real_cross_request_cache_hit():
    src = Path("panel/scripts/release-gate-python-app.py").read_text()

    assert "OpenAI cache warm turn1 response" in src
    assert "OpenAI cache warm turn2 response" in src
    assert "OpenAI cross-request cache hit" in src
    assert "cached_tokens_from_usage" in src


def test_live_engine_gate_cache_warm_prompt_crosses_paged_block_threshold():
    gate_module = _load_gate_module()

    assert len(gate_module.CACHE_WARM_PAD_TERMS) >= 96
    assert "cache-pad-095" in gate_module.CACHE_WARM_PROMPT
    assert "cobalt" in gate_module.CACHE_WARM_PROMPT


def test_packaged_console_script_shebang_gate_rejects_dev_paths(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    bin_dir = (
        tmp_path
        / "vMLX.app"
        / "Contents"
        / "Resources"
        / "bundled-python"
        / "python"
        / "bin"
    )
    bin_dir.mkdir(parents=True)
    script = bin_dir / "vmlx-engine"
    script.write_text(
        "#!/Users/example/mlx/vllm-mlx/panel/bundled-python/python/bin/python3\n"
        "import sys\n"
    )
    script.chmod(0o755)

    gate_module.check_packaged_console_script_shebangs(gate, tmp_path / "vMLX.app")

    assert gate.records[-1][0] == "packaged console-script shebangs"
    assert gate.records[-1][1] == "FAIL"
    assert "vmlx-engine" in gate.records[-1][2]


def test_packaged_console_script_shebang_gate_rejects_any_absolute_python_path(
    tmp_path,
):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    bin_dir = (
        tmp_path
        / "vMLX.app"
        / "Contents"
        / "Resources"
        / "bundled-python"
        / "python"
        / "bin"
    )
    bin_dir.mkdir(parents=True)
    script = bin_dir / "vmlx-engine"
    script.write_text(
        "#!/private/var/folders/build/bundled-python/python/bin/python3\n"
        "import sys\n"
    )
    script.chmod(0o755)

    gate_module.check_packaged_console_script_shebangs(gate, tmp_path / "vMLX.app")

    assert gate.records[-1][0] == "packaged console-script shebangs"
    assert gate.records[-1][1] == "FAIL"
    assert "vmlx-engine" in gate.records[-1][2]


def test_packaged_console_script_shebang_gate_accepts_relocatable_trampoline(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    bin_dir = (
        tmp_path
        / "vMLX.app"
        / "Contents"
        / "Resources"
        / "bundled-python"
        / "python"
        / "bin"
    )
    bin_dir.mkdir(parents=True)
    script = bin_dir / "vmlx-engine"
    script.write_text(
        "#!/bin/sh\n"
        "'''exec' \"$(dirname \"$0\")/python3\" -B -s \"$0\" \"$@\"\n"
        "' '''\n"
        "import sys\n"
    )
    script.chmod(0o755)

    gate_module.check_packaged_console_script_shebangs(gate, tmp_path / "vMLX.app")

    assert gate.records[-1][0] == "packaged console-script shebangs"
    assert gate.records[-1][1] == "PASS"


def test_packaged_signature_gate_rejects_ad_hoc_signature():
    gate_module = _load_gate_module()
    gate = _FakeGate("Executable=/tmp/vMLX.app/Contents/MacOS/vMLX\nSignature=adhoc\n")

    gate_module.check_packaged_developer_id_signature(
        gate,
        Path("/tmp/vMLX.app"),
        expected_team_id="55KGF2S5AY",
    )

    assert gate.records[-1][0] == "packaged Developer ID signature"
    assert gate.records[-1][1] == "FAIL"
    assert "ad-hoc" in gate.records[-1][2]


def test_packaged_signature_gate_accepts_expected_developer_id():
    gate_module = _load_gate_module()
    gate = _SequenceGate(
        {
            "packaged signature details": _developer_id_signature_stdout(),
            "packaged entitlements details": _release_entitlements_stdout(),
        }
    )

    gate_module.check_packaged_developer_id_signature(
        gate,
        Path("/tmp/vMLX.app"),
        expected_team_id="55KGF2S5AY",
    )

    assert gate.records[-1] == (
        "packaged Developer ID signature",
        "PASS",
        "team=55KGF2S5AY",
    )
    assert gate.run_calls[-1][0] == "packaged entitlements details"


def test_packaged_signature_gate_rejects_missing_release_entitlements():
    gate_module = _load_gate_module()
    gate = _SequenceGate(
        {
            "packaged signature details": _developer_id_signature_stdout(),
            "packaged entitlements details": _release_entitlements_stdout(
                missing="com.apple.security.cs.allow-jit"
            ),
        }
    )

    gate_module.check_packaged_developer_id_signature(
        gate,
        Path("/tmp/vMLX.app"),
        expected_team_id="55KGF2S5AY",
    )

    assert gate.records[-1][0] == "packaged Developer ID signature"
    assert gate.records[-1][1] == "FAIL"
    assert "missing release entitlements" in gate.records[-1][2]
    assert "com.apple.security.cs.allow-jit" in gate.records[-1][2]


def test_packaged_signature_gate_rejects_developer_id_without_hardened_runtime():
    gate_module = _load_gate_module()
    gate = _FakeGate(
        "\n".join(
            [
                "Executable=/tmp/vMLX.app/Contents/MacOS/vMLX",
                "CodeDirectory v=20500 size=325 flags=0x0(none) hashes=4+3 location=embedded",
                "Authority=Developer ID Application: ShieldStack LLC (55KGF2S5AY)",
                "Authority=Developer ID Certification Authority",
                "Authority=Apple Root CA",
                "TeamIdentifier=55KGF2S5AY",
            ]
        )
    )

    gate_module.check_packaged_developer_id_signature(
        gate,
        Path("/tmp/vMLX.app"),
        expected_team_id="55KGF2S5AY",
    )

    assert gate.records[-1][0] == "packaged Developer ID signature"
    assert gate.records[-1][1] == "FAIL"
    assert "hardened runtime" in gate.records[-1][2]


def test_release_dmg_final_sign_preserves_hardened_runtime_entitlements():
    script = Path("panel/scripts/build-release-dmgs.sh").read_text()

    final_sign_block = script[
        script.index("finalize_release_app_signature()") : script.index("find_staged_app()")
    ]

    assert "--options" in final_sign_block
    assert "runtime" in final_sign_block
    assert "--entitlements" in final_sign_block
    assert "build/entitlements.mac.plist" in final_sign_block


def test_release_dmg_staging_uses_recursive_signer_before_final_audit():
    script = Path("panel/scripts/build-release-dmgs.sh").read_text()
    build_one = script[
        script.index("build_one()") : script.index('case "$REQUESTED_FLAVOR"')
    ]

    stage_idx = build_one.index("run_electron_builder_action --mac --dir")
    final_sign_idx = build_one.index(
        'finalize_release_app_signature "$app_path" "$RELEASE_CODESIGN_IDENTITY"'
    )

    assert stage_idx < final_sign_idx
    assert "CSC_IDENTITY_AUTO_DISCOVERY=false" not in build_one
    assert "inside-out Developer-ID signing" in build_one
    assert "sign_remaining_app_macho_leaves" in script
    assert "verify_release_macho_leaves" in script


def test_release_dmg_macho_audit_scans_full_tree_once_without_suffix_filtering():
    script = Path("panel/scripts/build-release-dmgs.sh").read_text()
    audit = script[
        script.index("verify_release_macho_leaves()") : script.index(
            "verify_release_signature_identity()"
        )
    ]

    assert "os.walk(root, followlinks=False)" in audit
    assert "os.path.islink(path) or not os.path.isfile(path)" in audit
    for magic in (
        "feedface",
        "cefaedfe",
        "feedfacf",
        "cffaedfe",
        "cafebabe",
        "bebafeca",
        "cafebabf",
        "bfbafeca",
    ):
        assert f'"{magic}"' in audit
    assert 'is_macho_file "$native_file"' not in audit
    assert (
        'capture_toolchain_action find "$app_path/Contents" -type f -print'
        not in audit
    )


def test_r19_release_toolchain_plan_is_sealed_before_bound_actions():
    script = Path("panel/scripts/build-release-dmgs.sh").read_text()
    writer = script[
        script.index("write_r19_toolchain_plan()") : script.index(
            "run_bound_release_action()"
        )
    ]

    replace_idx = writer.index("os.replace(temporary, output)")
    seal_idx = writer.index("os.chmod(output, 0o400)")
    digest_idx = writer.index("print(hashlib.sha256(encoded).hexdigest())")

    assert replace_idx < seal_idx < digest_idx


def test_r19_release_driver_plan_is_sealed_before_bound_actions():
    script = Path("panel/scripts/build-release-dmgs.sh").read_text()
    writer = script[
        script.index("write_r19_build_plan()") : script.index(
            "assert_r19_source_identity()"
        )
    ]

    replace_idx = writer.index("os.replace(temporary, output)")
    seal_idx = writer.index("os.chmod(output, 0o400)")
    digest_idx = writer.index("print(hashlib.sha256(encoded).hexdigest())")

    assert replace_idx < seal_idx < digest_idx


def test_r19_release_builder_reuses_consumed_v5_checks_without_rerunning_suites():
    script = Path("panel/scripts/build-release-dmgs.sh").read_text()
    start = script.index(
        'echo "==> Reinstalling exact panel dependencies from package-lock.json"'
    )
    end = script.index("\nfi\n\nis_macho_file()", start)
    production_gate = script[start:end]

    assert "run_complete_python_source_suite" not in script
    assert "run_toolchain_action npm ci" in production_gate
    assert "run_toolchain_action npm test" not in production_gate
    assert "run_toolchain_action npm run typecheck" not in production_gate
    assert (
        "Reusing exact-head V5 Python, panel, typecheck, and production-build evidence"
        in production_gate
    )
    assert (
        'assert_r19_source_identity "after exact-head V5 check reuse"'
        in production_gate
    )


def test_bundled_verifier_rejects_non_relocatable_console_shebangs():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    assert "check_console_script_shebangs" in verifier
    assert '[[ "$first_line" == \'#!\'*python* ]]' in verifier
    assert "/Applications/vMLX.app" in verifier
    assert "non-relocatable console-script shebangs" in verifier


def test_bundle_python_relocates_local_source_console_scripts_after_install():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()
    local_install_idx = bundler.index('echo "==> Installing vmlx-engine + jang_tools')
    cleanup_idx = bundler.index("# Clean up to reduce size", local_install_idx)
    local_install_block = bundler[local_install_idx:cleanup_idx]

    assert 'for SCRIPT in "$BUNDLE_DIR/python/bin/"vmlx* "$BUNDLE_DIR/python/bin/"jang*' in local_install_block
    assert '\\$(dirname \\"\\$0\\")/python3' in local_install_block
    assert '-B -s \\"\\$0\\" \\"\\$@\\"' in local_install_block
    assert '[[ "$FIRST_LINE" == \'#!\'*python* ]]' in local_install_block


def test_bundle_python_preserves_release_gate_evidence_directory():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()

    assert 'rm -rf "$VMLX_LOCAL/build"' not in bundler
    assert '"$VMLX_LOCAL/build/lib"' in bundler
    assert '"$VMLX_LOCAL/build"/bdist.*' in bundler
    assert '"$VMLX_LOCAL/build"/temp.*' in bundler
    assert "tracked release-gate evidence" in bundler


def test_bundle_python_preserves_packaged_engine_runtime_diagnostics():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()

    assert 'find "$SITE" -type d -name "tests"' in bundler
    assert '! -path "$SITE/vmlx_engine/tests"' in bundler
    assert "runtime diagnostics are explicit" in bundler


def test_bundle_python_rejects_dirty_vmlx_package_source_by_default():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()

    assert "check_local_vmlx_source_clean" in bundler
    assert "VMLX_ALLOW_DIRTY_SOURCE" in bundler
    assert "vMLX package source is dirty" in bundler
    assert "panel/scripts panel/src" in bundler


def test_bundle_python_uses_one_pinned_binary_opencv_distribution():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()
    pyproject = Path("pyproject.toml").read_text()

    assert 'OPENCV_VERSION="4.13.0.92"' in bundler
    assert 'MLX_AUDIO_VERSION="0.4.6"' in bundler
    assert '"opencv-python==$OPENCV_VERSION"' in bundler
    assert '"mlx-audio==$MLX_AUDIO_VERSION"' in bundler
    assert "opencv-python-headless" not in bundler
    assert '"$PYTHON" -m pip install --only-binary=:all:' in bundler
    assert 'rm -rf "$SITE/setuptools"' not in bundler
    assert "setuptools: KEEP" in bundler
    assert '"opencv-python==4.13.0.92"' in pyproject
    assert "opencv-python-headless" not in pyproject
    assert 'version("opencv-python")' in verifier
    assert 'version("opencv-python-headless")' in verifier
    assert "run_bundled_python -m pip check" in verifier


def test_bundle_python_isolates_host_python_and_publishes_only_verified_staging():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    assert "export PYTHONNOUSERSITE=1" in bundler
    assert "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV" in bundler
    assert 'BUNDLE_DIR="$PANEL_DIR/.bundled-python.staging.$$"' in bundler
    assert 'PREVIOUS_BUNDLE_DIR="$PANEL_DIR/.bundled-python.previous.$$"' in bundler
    verify_idx = bundler.index('VMLX_BUNDLED_PYTHON_DIR="$BUNDLE_DIR"')
    backup_idx = bundler.index(
        'mv "$FINAL_BUNDLE_DIR" "$PREVIOUS_BUNDLE_DIR"',
        verify_idx,
    )
    publish_idx = bundler.index(
        'mv "$BUNDLE_DIR" "$FINAL_BUNDLE_DIR"',
        backup_idx,
    )
    assert verify_idx < backup_idx < publish_idx
    assert 'mv "$PREVIOUS_BUNDLE_DIR" "$FINAL_BUNDLE_DIR" || true' in bundler
    assert 'BUNDLE_ROOT="${VMLX_BUNDLED_PYTHON_DIR:-$PANEL/bundled-python}"' in verifier
    assert "export PYTHONNOUSERSITE=1" in verifier
    assert "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV" in verifier


def test_bundle_python_retries_only_owned_tree_cleanup_and_still_fails_closed():
    bundler = Path("panel/scripts/bundle-python.sh").read_text()

    assert "remove_bundle_tree_with_retry()" in bundler
    assert "for attempt in 1 2 3" in bundler
    assert '/bin/rm -rf "$target"' in bundler
    assert "retrying transient bundled-Python cleanup" in bundler
    assert "cleanup did not converge after 3 attempts" in bundler
    assert 'remove_bundle_tree_with_retry "$BUNDLE_DIR"' in bundler
    assert 'remove_bundle_tree_with_retry "$PREVIOUS_BUNDLE_DIR"' in bundler
    assert 'rm -rf "$PREVIOUS_BUNDLE_DIR"' not in bundler


def test_machine_specific_apple_notary_and_signing_helpers_are_ignored():
    private_helpers = (
        "panel/scripts/apple-notary-profile.sh",
        "panel/scripts/vmlx-notary-auth.py",
        "panel/scripts/release-signing-profile.zsh",
        "scripts/check-vmlx-notary-profile.exp",
    )

    for helper in private_helpers:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", helper],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, helper


def test_release_gate_uses_anthropic_native_thinking_disable():
    gate_module = _load_gate_module()

    body = gate_module.apply_anthropic_thinking(
        {"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        "off",
    )

    assert body["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in body


def test_release_gate_uses_anthropic_native_thinking_enable():
    gate_module = _load_gate_module()

    body = gate_module.apply_anthropic_thinking(
        {"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        "on",
    )

    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] > 0


def test_packaged_bundled_version_parity_fails_on_stale_bundled_engine():
    gate_module = _load_gate_module()
    gate = _FakeGate("1.5.23\n")

    gate_module.check_packaged_bundled_import_version(
        gate, Path("/app/python3"), "1.5.25", "1.5.25"
    )

    assert gate.records[-1] == (
        "packaged bundled version",
        "FAIL",
        "app=1.5.25, bundled=1.5.23, expected=1.5.25",
    )


def test_packaged_bundled_content_gate_rejects_removed_dsv4_force_flags(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    engine_dir = tmp_path / "vmlx_engine"
    engine_dir.mkdir()
    (engine_dir / "server.py").write_text(
        "os.environ.get('VMLX_DSV4_ALLOW_CHAT', '0')\n"
    )

    gate_module.check_no_removed_env_var_force_flips(gate, engine_dir)

    assert gate.records[-1][0] == "bundled removed env-var gate"
    assert gate.records[-1][1] == "FAIL"
    assert "VMLX_DSV4_ALLOW_CHAT" in gate.records[-1][2]


def test_packaged_bundled_content_gate_passes_clean_engine_tree(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    engine_dir = tmp_path / "vmlx_engine"
    engine_dir.mkdir()
    (engine_dir / "server.py").write_text("DSV4_COMPOSITE_CACHE = True\n")

    gate_module.check_no_removed_env_var_force_flips(gate, engine_dir)

    assert gate.records[-1] == (
        "bundled removed env-var gate",
        "PASS",
        str(engine_dir),
    )


def test_packaged_bundled_server_hash_gate_fails_on_content_drift(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    source_dir = tmp_path / "source" / "vmlx_engine"
    bundled_dir = tmp_path / "bundled" / "vmlx_engine"
    source_dir.mkdir(parents=True)
    bundled_dir.mkdir(parents=True)
    (source_dir / "server.py").write_text("CURRENT = True\n")
    (bundled_dir / "server.py").write_text("STALE = True\n")

    gate_module.check_bundled_source_file_hashes(
        gate, bundled_dir, source_dir=source_dir, rel_paths=("server.py",)
    )

    assert gate.records[-1][0] == "bundled source content hash"
    assert gate.records[-1][1] == "FAIL"
    assert "server.py" in gate.records[-1][2]


def test_packaged_bundled_server_hash_gate_passes_on_matching_content(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    source_dir = tmp_path / "source" / "vmlx_engine"
    bundled_dir = tmp_path / "bundled" / "vmlx_engine"
    source_dir.mkdir(parents=True)
    bundled_dir.mkdir(parents=True)
    (source_dir / "server.py").write_text("CURRENT = True\n")
    (bundled_dir / "server.py").write_text("CURRENT = True\n")

    gate_module.check_bundled_source_file_hashes(
        gate, bundled_dir, source_dir=source_dir, rel_paths=("server.py",)
    )

    assert gate.records[-1] == (
        "bundled source content hash",
        "PASS",
        "server.py",
    )


def test_packaged_bundled_hash_gate_covers_runtime_files_changed_for_release():
    gate_module = _load_gate_module()

    expected = {
        "__init__.py",
        "server.py",
        "api/utils.py",
        "api/anthropic_adapter.py",
        "api/ollama_adapter.py",
        "block_disk_store.py",
        "cli.py",
        "disk_cache.py",
        "engine/batched.py",
        "loaders/load_jangtq_dsv4.py",
        "mllm_batch_generator.py",
        "mllm_scheduler.py",
        "model_configs.py",
        "model_config_registry.py",
        "speculative.py",
        "models/llm.py",
        "models/mllm.py",
        "models/step3p7_mlx_vlm.py",
        "omni_multimodal.py",
        "paged_cache.py",
        "prefix_cache.py",
        "runtime_patches/gemma4_vision.py",
        "runtime_patches/gemma4_processing.py",
        "scheduler.py",
        "patches/mlx_vlm_mtp/qwen35_vl.py",
        "utils/single_batch_generator.py",
        "utils/head_dim_detection.py",
        "utils/mlx_vlm_compat.py",
        "utils/ssm_companion_cache.py",
        "utils/ssm_companion_disk_store.py",
        "utils/jang_loader.py",
        "utils/nanbeige_runtime.py",
        "utils/tokenizer.py",
    }

    assert expected.issubset(set(gate_module.BUNDLED_SOURCE_HASH_PATHS))


def test_packaged_bundled_hash_gate_covers_critical_jang_tools_files():
    gate_module = _load_gate_module()

    expected = {
        "capabilities.py",
        "convert.py",
        "convert_hy3_jangtq.py",
        "loader.py",
        "load_jangtq.py",
        "load_jangtq_vlm.py",
        "load_jangtq_kimi_vlm.py",
        "dsv4/mlx_model.py",
        "dsv4/pool_quant_cache.py",
        "hy3/__init__.py",
        "hy3/model.py",
        "hy3/runtime.py",
        "laguna/runtime.py",
        "kimi_prune/generate_vl.py",
        "kimi_prune/runtime_patch.py",
        "mimo_v2/mlx_model.py",
        "nanbeige/__init__.py",
        "nanbeige/model.py",
        "nanbeige/mlx_register.py",
        "step37/__init__.py",
        "step37/nvfp4_codec.py",
        "step37/step3p7_mlx.py",
        "topk_override.py",
        "turboquant/fused_gate_up_kernel.py",
        "turboquant/gather_tq_kernel.py",
        "turboquant/hadamard_kernel.py",
        "turboquant/mpp_nax_kernel.py",
        "turboquant/tq_kernel.py",
    }

    assert expected.issubset(set(gate_module.JANG_TOOLS_SOURCE_HASH_PATHS))


def test_release_gate_objective_digest_allows_deferred_open_requirement(tmp_path):
    gate_module = _load_gate_module()
    digest = tmp_path / "objective.json"
    digest.write_text(
        json.dumps(
            {
                "requirements": [
                    {"requirement": "safe cache", "status": "pass"},
                    {"requirement": "DSV4 long-output/code/file-generation quality is release-cleared", "status": "open"},
                ]
            }
        )
    )
    gate = _FakeGate("refreshed\n")

    gate_module.check_objective_proof_digest(gate, digest_path=digest)

    assert gate.records[-1] == (
        "objective proof digest",
        "PASS",
        f"{digest}; deferred=1",
    )


def test_release_gate_objective_digest_fails_on_non_deferred_open_requirement(tmp_path):
    gate_module = _load_gate_module()
    digest = tmp_path / "objective.json"
    digest.write_text(
        json.dumps(
            {
                "requirements": [
                    {"requirement": "safe cache", "status": "pass"},
                    {"requirement": "Qwen video MTP is release-cleared", "status": "open"},
                ]
            }
        )
    )
    gate = _FakeGate("refreshed\n")

    gate_module.check_objective_proof_digest(gate, digest_path=digest)

    assert gate.records[-1] == (
        "objective proof digest",
        "FAIL",
        "Qwen video MTP is release-cleared",
    )


def test_release_gate_static_runs_objective_digest_gate():
    src = Path("panel/scripts/release-gate-python-app.py").read_text()

    assert "def check_objective_proof_digest" in src
    assert "check_objective_proof_digest(gate)" in src


def test_release_gate_static_requires_release_ready_manifest():
    src = Path("panel/scripts/release-gate-python-app.py").read_text()

    assert "def check_release_ready_manifest" in src
    assert "check_release_ready_manifest(gate)" in src
    assert "--skip-release-manifest" in src
    assert "--require-release-ready" in src
    assert "run_release_regression_manifest.py" in src


def test_packaged_bundled_package_hash_gate_fails_on_content_drift(tmp_path):
    gate_module = _load_gate_module()
    gate = _FakeGate("")
    source_dir = tmp_path / "source" / "jang_tools"
    bundled_dir = tmp_path / "bundled" / "jang_tools"
    source_dir.mkdir(parents=True)
    bundled_dir.mkdir(parents=True)
    (source_dir / "load_jangtq.py").write_text("CURRENT = True\n")
    (bundled_dir / "load_jangtq.py").write_text("STALE = True\n")

    gate_module.check_bundled_package_file_hashes(
        gate,
        "jang_tools",
        bundled_dir,
        source_dir,
        rel_paths=("load_jangtq.py",),
    )

    assert gate.records[-1][0] == "bundled jang_tools content hash"
    assert gate.records[-1][1] == "FAIL"
    assert "load_jangtq.py" in gate.records[-1][2]


def test_electron_builder_runs_bundled_python_gate_before_packaging():
    pkg = json.loads(Path("panel/package.json").read_text())
    hook = pkg["build"].get("beforePack")
    assert hook == "scripts/electron-builder-before-pack.cjs"

    hook_src = Path("panel/scripts/electron-builder-before-pack.cjs").read_text()
    assert "verify-bundled-python.sh" in hook_src
    assert "electron-vite" in hook_src
    assert "VMLX_BEFORE_PACK_SKIP_VITE" in hook_src
    assert "require.main === module" in hook_src


def test_electron_builder_dmg_contains_only_current_electron_app():
    pkg = json.loads(Path("panel/package.json").read_text())
    contents = pkg["build"]["dmg"]["contents"]

    assert contents == [
        {"x": 160, "y": 220, "name": "vMLX.app"},
        {"x": 600, "y": 220, "type": "link", "path": "/Applications"},
    ]
    assert "vMLX 2 (beta).app" not in json.dumps(pkg)
    assert "build/extra" not in json.dumps(pkg)


def test_verify_bundled_python_blocks_removed_dsv4_force_flags():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    assert "VMLX_DSV4_ALLOW_CHAT" in verifier
    assert "VMLX_DSV4_ALLOW_THINKING" in verifier
    assert "VMLX_DSV4_FORCE_DIRECT_RAIL" in verifier
    assert "RELEASE BLOCKED — bundled-python contains removed DSV4 env-var force-flips" in verifier


def test_verify_bundled_python_hash_gate_covers_release_runtime_files():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    expected_engine_files = {
        "server.py",
        "api/utils.py",
        "api/anthropic_adapter.py",
        "api/ollama_adapter.py",
        "block_disk_store.py",
        "cli.py",
        "disk_cache.py",
        "engine/batched.py",
        "loaders/load_jangtq_dsv4.py",
        "mllm_batch_generator.py",
        "mllm_scheduler.py",
        "model_configs.py",
        "model_config_registry.py",
        "speculative.py",
        "models/llm.py",
        "models/mllm.py",
        "models/step3p7_mlx_vlm.py",
        "omni_multimodal.py",
        "paged_cache.py",
        "prefix_cache.py",
        "runtime_patches/gemma4_processing.py",
        "scheduler.py",
        "utils/single_batch_generator.py",
        "utils/head_dim_detection.py",
        "utils/ssm_companion_cache.py",
        "utils/ssm_companion_disk_store.py",
        "utils/jang_loader.py",
        "utils/nanbeige_runtime.py",
        "utils/tokenizer.py",
    }
    expected_jang_tools_files = {
        "capabilities.py",
        "convert.py",
        "convert_hy3_jangtq.py",
        "loader.py",
        "load_jangtq.py",
        "load_jangtq_kimi_vlm.py",
        "hy3/__init__.py",
        "hy3/model.py",
        "hy3/runtime.py",
        "laguna/runtime.py",
        "kimi_prune/generate_vl.py",
        "kimi_prune/runtime_patch.py",
        "step37/__init__.py",
        "step37/nvfp4_codec.py",
        "step37/step3p7_mlx.py",
        "topk_override.py",
        "turboquant/fused_gate_up_kernel.py",
        "turboquant/gather_tq_kernel.py",
        "turboquant/hadamard_kernel.py",
        "turboquant/mpp_nax_kernel.py",
        "turboquant/tq_kernel.py",
    }

    for rel in expected_engine_files | expected_jang_tools_files:
        assert f'"{rel}"' in verifier

    assert "cd /tmp" in verifier
    assert "PYTHONPATH=" in verifier


def test_verify_bundled_python_import_gate_covers_hy3_jangtq_runtime_modules():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    for mod in (
        "jang_tools.hy3",
        "jang_tools.hy3.runtime",
        "jang_tools.mimo_v2.mlx_register",
        "jang_tools.topk_override",
        "jang_tools.capabilities",
    ):
        assert f'("{mod}",' in verifier
    assert '("mlx_lm.models.mimo_v2",' in verifier


def test_verify_bundled_python_gates_nanbeige_content_and_registration_order():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    for rel in (
        "utils/nanbeige_runtime.py",
        "nanbeige/__init__.py",
        "nanbeige/model.py",
        "nanbeige/mlx_register.py",
    ):
        assert f'"{rel}"' in verifier

    imports = (
        "jang_tools.nanbeige",
        "jang_tools.nanbeige.model",
        "jang_tools.nanbeige.mlx_register",
        "mlx_lm.models.nanbeige",
    )
    positions = []
    for module in imports:
        marker = f'("{module}",'
        assert marker in verifier
        positions.append(verifier.index(marker))
    assert positions == sorted(positions)
    assert "Nanbeige mlx-lm registration missing" in verifier


def test_verify_bundled_python_checks_laguna_mixed_affine_runtime_contract():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    assert '"laguna/runtime.py"' in verifier
    assert '("jang_tools.laguna.runtime",' in verifier
    assert "LAGUNA_MIXED_AFFINE_RUNTIME_VERSION" in verifier
    assert "infer_affine_bits_from_shapes" in verifier
    assert "(100352, 384)" in verifier
    assert "(100352, 32)" in verifier
    assert "_laguna_bits != 6" in verifier


def test_verify_bundled_python_import_gate_covers_step37_source_runtime():
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    assert '("jang_tools.step37.step3p7_mlx", "jang_tools.step37.step3p7_mlx"' in verifier
    assert '("vmlx_engine.models.step3p7_mlx_vlm", "vmlx_engine Step3p7 VLM runtime"' in verifier
    assert "_register_step3p7_mlx_vlm_runtime()" in verifier
    assert '"mlx_vlm.models.step3p7"' in verifier
    assert '"mlx_vlm.models.step3p7.processing_step3"' in verifier
    assert "Step3p7 source VLM runtime missing" in verifier
    assert "Step3p7 mlx-vlm registration missing" in verifier


def test_nemotron_omni_media_dependency_timm_is_packaged_and_verified():
    pyproject = Path("pyproject.toml").read_text()
    bundle_script = Path("panel/scripts/bundle-python.sh").read_text()
    verifier = Path("panel/scripts/verify-bundled-python.sh").read_text()

    assert '"timm>=1.0.20"' in pyproject
    assert '"einops>=0.8.0"' in pyproject
    assert '"librosa>=0.10.0"' in pyproject
    assert '"timm>=1.0.20"' in bundle_script
    assert '"einops>=0.8.0"' in bundle_script
    assert 'librosa sounddevice miniaudio pyloudnorm numba' in bundle_script
    assert '("timm", "timm vision backbone"' in verifier
    assert '("einops", "einops tensor rearrange"' in verifier
    assert '("librosa", "librosa audio features"' in verifier


def test_electron_builder_before_pack_hook_runs_verifier_in_direct_smoke(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    verifier = scripts / "verify-bundled-python.sh"
    verifier.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok > \"$PWD/verify-ran\"\n")
    verifier.chmod(0o755)

    env = dict(os.environ)
    env["VMLX_BEFORE_PACK_SKIP_VITE"] = "1"
    proc = subprocess.run(
        ["node", str(Path("panel/scripts/electron-builder-before-pack.cjs").resolve())],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "verify-ran").read_text() == "ok\n"
    assert "skipped electron-vite build" in proc.stdout


def test_completion_attestation_cleanup_retries_transient_directory_races():
    hook_src = Path("panel/scripts/electron-builder-before-pack.cjs").read_text()

    assert "rmSync(extracted, {" in hook_src
    assert "recursive: true" in hook_src
    assert "force: true" in hook_src
    assert "maxRetries: 8" in hook_src
    assert "retryDelay: 100" in hook_src


def test_electron_builder_before_pack_hook_rejects_skip_vite_in_pack_context(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"version": "1.6.17"}),
        encoding="utf-8",
    )
    verifier = scripts / "verify-bundled-python.sh"
    verifier.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok > \"$PWD/verify-ran\"\n")
    verifier.chmod(0o755)

    hook_path = Path("panel/scripts/electron-builder-before-pack.cjs").resolve()
    js = (
        "process.env.VMLX_BEFORE_PACK_SKIP_VITE = '1';"
        f"const hook = require({json.dumps(str(hook_path))});"
        f"hook({{packager: {{projectDir: {json.dumps(str(tmp_path))}}}}})"
        ".then(() => process.exit(0))"
        ".catch((err) => { console.error(err.message); process.exit(3); });"
    )
    proc = subprocess.run(
        ["node", "-e", js],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 3
    assert (tmp_path / "verify-ran").read_text() == "ok\n"
    assert "only allowed for direct hook smoke tests" in proc.stderr


def test_electron_builder_before_pack_rejects_direct_r19_packaging_before_verifier(
    tmp_path,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "version": "1.6.19",
                "build": {
                    "mac": {
                        "notarize": {
                            "teamId": "55KGF2S5AY",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    verifier = scripts / "verify-bundled-python.sh"
    verifier.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\necho ran > \"$PWD/verify-ran\"\n",
        encoding="utf-8",
    )
    verifier.chmod(0o755)

    hook_path = Path("panel/scripts/electron-builder-before-pack.cjs").resolve()
    js = (
        f"const hook = require({json.dumps(str(hook_path))});"
        f"hook({{packager: {{projectDir: {json.dumps(str(tmp_path))}}}}})"
        ".then(() => process.exit(0))"
        ".catch((err) => { console.error(err.message); process.exit(3); });"
    )
    env = dict(os.environ)
    for name in (
        "VMLX_RELEASE_SCOPE",
        "VMLX_R19_OFFICIAL_PACKAGING",
        "VMLX_R19_EXPECTED_TEAM_ID",
        "VMLX_R19_EXPECTED_CODESIGN_IDENTITY",
        "CSC_NAME",
    ):
        env.pop(name, None)
    proc = subprocess.run(
        ["node", "-e", js],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 3
    assert "requires VMLX_RELEASE_SCOPE=r19_production" in proc.stderr
    assert not (tmp_path / "verify-ran").exists()


def test_r19_release_builder_rejects_single_flavor_python_override_and_wrong_team(
    tmp_path,
):
    panel = tmp_path / "panel"
    scripts = panel / "scripts"
    node_modules = panel / "node_modules"
    venv_bin = tmp_path / ".venv" / "bin"
    scripts.mkdir(parents=True)
    node_modules.mkdir()
    venv_bin.mkdir(parents=True)
    builder = scripts / "build-release-dmgs.sh"
    builder.write_text(
        Path("panel/scripts/build-release-dmgs.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    builder.chmod(0o755)
    (scripts / "release-python-action.cjs").write_text(
        Path("panel/scripts/release-python-action.cjs").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    release_python = venv_bin / "python"
    release_python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    release_python.chmod(0o755)
    (panel / "package.json").write_text(
        json.dumps(
            {
                "version": "1.6.19",
                "build": {
                    "mac": {
                        "notarize": {
                            "teamId": "55KGF2S5AY",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    base_env = dict(os.environ)
    for name in (
        "PYTHON",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "VMLX_RELEASE_CODESIGN_IDENTITY",
        "VMLINUX_RELEASE_CODESIGN_IDENTITY",
        "CSC_NAME",
    ):
        base_env.pop(name, None)
    base_env["VMLX_RELEASE_SCOPE"] = "r19_production"

    single = subprocess.run(
        [str(builder), "sequoia"],
        cwd=panel,
        env=base_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert single.returncode == 1
    assert "must build both Sequoia and Tahoe via flavor=all" in single.stderr

    generic_scope_env = dict(base_env)
    generic_scope_env["VMLX_RELEASE_SCOPE"] = "production"
    generic_single = subprocess.run(
        [str(builder), "sequoia"],
        cwd=panel,
        env=generic_scope_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generic_single.returncode == 1
    assert "must build both Sequoia and Tahoe via flavor=all" in generic_single.stderr

    python_override_env = dict(base_env)
    python_override_env["PYTHON"] = "/usr/bin/python3"
    python_override = subprocess.run(
        [str(builder), "all"],
        cwd=panel,
        env=python_override_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert python_override.returncode == 1
    assert "cannot be overridden by PYTHON" in python_override.stderr

    wrong_team_env = dict(base_env)
    wrong_team_env["VMLX_RELEASE_CODESIGN_IDENTITY"] = (
        "Developer ID Application: Other Team (AAAAAAAAAA)"
    )
    wrong_team = subprocess.run(
        [str(builder), "all"],
        cwd=panel,
        env=wrong_team_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_team.returncode == 1
    assert "requires Developer ID Application: ShieldStack LLC" in wrong_team.stderr

    package = json.loads((panel / "package.json").read_text(encoding="utf-8"))
    package["version"] = "1.6.17"
    (panel / "package.json").write_text(json.dumps(package), encoding="utf-8")
    wrong_version = subprocess.run(
        [str(builder), "all"],
        cwd=panel,
        env=base_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_version.returncode == 1
    assert (
        "VMLX_RELEASE_SCOPE=r19_production requires package version 1.6.19"
        in wrong_version.stderr
    )
    generic_wrong_version = subprocess.run(
        [str(builder), "all"],
        cwd=panel,
        env=generic_scope_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generic_wrong_version.returncode == 1
    assert (
        "public production packaging is not implemented for package version 1.6.17"
        in generic_wrong_version.stderr
    )

    package["version"] = "1.6.19"
    (panel / "package.json").write_text(json.dumps(package), encoding="utf-8")
    release_python.unlink()
    release_python.symlink_to(sys._base_executable)
    foreign_venv = subprocess.run(
        [str(builder), "all"],
        cwd=panel,
        env=base_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert foreign_venv.returncode == 1
    assert "pyvenv.cfg" in foreign_venv.stderr
