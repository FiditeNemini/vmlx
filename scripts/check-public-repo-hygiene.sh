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
      /^autoresearch\// ||
      /^tmp\// ||
      /^build\// ||
      /^assets\/tools-tab\.png$/ ||
      /^panel\/(CHANGELOG|ENGINE-UPDATES|PROJECT|SETUP)\.md$/ ||
      /^productionapp\/(INSTALL|TECHNICAL-NOTES)\.md$/ ||
      /^vmlx_engine\/docs\/CODEBOOK-DEVELOPMENT\.md$/ ||
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

private_host_pattern='erics-m5-'"max"'([0-9]*)?([.]tail[[:alnum:]]+[.]ts[.]net|[.]local)'
private_volume_pattern='/Volumes/'"Erics"'LLMDrive'
private_cache_pattern='[.]cache/vmlx-'"proof"

private_strings=$(
  {
    git grep -Il -E \
      "(${private_host_pattern}|${private_volume_pattern}|${private_cache_pattern})" \
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

invalid_commit_emails=$(
  git log --all --format='%ae%n%ce' |
    awk '$0 !~ /^[^[:space:]<>@]+@[^[:space:]<>@]+$/ {print}' |
    sort -u
)

if [ -n "$invalid_commit_emails" ]; then
  printf '%s\n' \
    'ERROR: public repository history contains malformed author or committer email metadata.' >&2
  exit 1
fi

private_message_commits=$(
  git log --all --format='@@%H%n%B' |
    awk \
      -v private_host_pattern="$private_host_pattern" \
      -v private_volume_pattern="$private_volume_pattern" \
      -v private_cache_pattern="$private_cache_pattern" '
      /^@@[0-9a-f]+$/ {
        commit = substr($0, 3)
        next
      }
      {
        line = $0
        while (match(line, /\/Users\/[A-Za-z0-9._-]+/)) {
          candidate = substr(line, RSTART, RLENGTH)
          if (candidate != "/Users/example" && candidate != "/Users/u") {
            bad[commit] = 1
          }
          line = substr(line, RSTART + RLENGTH)
        }
        if (($0 ~ private_host_pattern) || ($0 ~ private_volume_pattern) || ($0 ~ private_cache_pattern)) {
          bad[commit] = 1
        }
      }
      END {
        for (commit in bad) {
          print commit
        }
      }
    ' |
    sort
)

if [ -n "$private_message_commits" ]; then
  printf '%s\n' \
    'ERROR: public history contains private identifiers in commit messages:' \
    "$private_message_commits" >&2
  exit 1
fi

printf '%s\n' 'Public repository hygiene check passed.'
