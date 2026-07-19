import importlib

from tests.cross_matrix.output_counts import parse_counts, parse_vitest_counts


COUNT_PARSER_RUNNERS = (
    "run_api_surface_contract",
    "run_cache_architecture_contract",
    "run_generation_defaults_contract",
    "run_max_output_context_contract",
    "run_mcp_policy_contract",
    "run_model_artifact_format_contract",
    "run_model_family_detection_contract",
    "run_native_mtp_contract",
    "run_noheavy_api_cache_contract",
    "run_noheavy_panel_settings_contract",
    "run_packaged_integrity_contract",
    "run_parser_registry_contract",
    "run_reasoning_template_contract",
    "run_tool_call_contract",
    "run_vl_media_cache_contract",
)


def test_vitest_ansi_summary_uses_test_count_not_test_file_count():
    output = (
        "\x1b[2m Test Files \x1b[22m \x1b[1m\x1b[32m9 passed\x1b[39m\x1b[22m\n"
        "\x1b[2m      Tests \x1b[22m \x1b[1m\x1b[32m640 passed\x1b[39m\x1b[22m"
        "\x1b[2m | \x1b[22m\x1b[33m3 skipped\x1b[39m\n"
    )

    assert parse_counts(output) == {
        "passed": 640,
        "skipped": 3,
        "deselected": None,
    }
    assert parse_vitest_counts(output) == {
        "test_files_passed": 9,
        "tests_passed": 640,
        "tests_skipped": 3,
    }


def test_pytest_and_nested_runner_counts_remain_supported():
    assert parse_counts("5942 passed, 96 skipped, 261 deselected") == {
        "passed": 5942,
        "skipped": 96,
        "deselected": 261,
    }
    assert parse_counts("passed=42 skipped=None deselected=7") == {
        "passed": 42,
        "skipped": None,
        "deselected": 7,
    }


def test_every_count_parser_consumer_hashes_the_shared_parser_source():
    for name in COUNT_PARSER_RUNNERS:
        module = importlib.import_module(f"tests.cross_matrix.{name}")
        assert "tests/cross_matrix/output_counts.py" in module.SOURCE_HASH_FILES
