#!/bin/sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

forbidden=$(
  git ls-files |
    awk '
      /^docs\// &&
        !/^docs\/ARCHITECTURE\.md$/ &&
        !/^docs\/index\.md$/ &&
        !/^docs\/mlxstudio-releases-readme\.md$/ &&
        !/^docs\/api\// &&
        !/^docs\/benchmarks\// &&
        !/^docs\/development\/(architecture|build-test-deploy|contributing)\.md$/ &&
        !/^docs\/getting-started\// &&
        !/^docs\/guides\// &&
        !/^docs\/reference\// {
          print
          next
        }
      index($0, "/") == 0 &&
        /\.md$/ &&
        !/^(README|CHANGELOG|CONTRIBUTING|SECURITY|CODE_OF_CONDUCT)\.md$/ {
          print
          next
        }
      /^notes\// ||
      /^PLANS\// ||
      /^build\// ||
      /(^|\/)node_modules\// ||
      /^panel\/docs\/plans\// ||
      /^tests\/e2e\/results\// ||
      /^tests\/e2e\/panel-driver\/node_modules\// ||
      /^tests\/benchmark\/outputs\// ||
      /^tests\/e2e\/(AUDIT-REPORT|MATRIX|UI-SUITE)\.md$/ ||
      /^nohup\.out$/ ||
      /^trace_err2?\.txt$/ ||
      /^(gsm8k_qwen3_0\.6b_results|vlm_benchmark_results)\.json$/ ||
      /^vmlx_engine\/models\/minimax_m3\/(BUILD-STATUS|MASTER-STATUS|CAMPAIGN-CHECKLIST|CAMPAIGN-PROGRESS-LOG|M3-EAGLE3-NATIVE-MTP-HANDOFF|M3-MOE-QUANT-FIX-HANDOFF)\.md$/ ||
      /^vmlx_engine\/models\/minimax_m3\/MODEL-MATRIX-AUTODETECT\.txt$/ ||
      /^\.agents\// ||
      /^\.agent\// ||
      /^\.claude\// ||
      /^\.codex\// ||
      /^\.sisyphus\// ||
      /^\.factory\// ||
      /(^|\/)(screenshots?|screen-recordings?|cdp-captures?|raw-sse|runtime-logs?)(\/|$)/ {
        print
      }
    '
)

if [ -n "$forbidden" ]; then
  printf '%s\n' \
    'ERROR: public repository contains forbidden private/internal artifacts:' \
    "$forbidden" >&2
  exit 1
fi

private_volume_pattern='/Volumes/'"Erics"'LLMDrive'

private_strings=$(
  {
    git grep -Il -E \
      "(test-host([0-9]*)?(\.tail[[:alnum:]]+\.ts\.net|\.local)|${private_volume_pattern})" \
      -- . ':(exclude)scripts/check-public-repo-hygiene.sh' 2>/dev/null || true
    git grep -InE '/Users/[A-Za-z0-9._-]+' -- . 2>/dev/null |
      grep -Ev '/Users/(example|u)(/|[^A-Za-z0-9._-])' |
      grep -v '^scripts/check-public-repo-hygiene.sh:' |
      awk -F: '{print $1}' || true
  } | sort -u
)

if [ -n "$private_strings" ]; then
  printf '%s\n' \
    'ERROR: public repository contains private host or filesystem paths:' \
    "$private_strings" >&2
  exit 1
fi

printf '%s\n' 'Public repository hygiene check passed.'
