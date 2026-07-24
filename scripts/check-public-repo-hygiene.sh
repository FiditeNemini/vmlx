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
      /^panel\/docs\/plans\// ||
      /^tests\/e2e\/results\// ||
      /^tests\/e2e\/(AUDIT-REPORT|MATRIX|UI-SUITE)\.md$/ ||
      /^\.agents\// ||
      /^\.agent\// ||
      /^\.claude\// ||
      /^\.codex\// ||
      /^\.sisyphus\// ||
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

printf '%s\n' 'Public repository hygiene check passed.'
