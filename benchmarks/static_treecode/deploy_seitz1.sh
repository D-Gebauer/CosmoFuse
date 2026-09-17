#!/usr/bin/env bash
# Sync the working tree to a separate directory on seitz1 (the user's own
# ~/research/CosmoFuse checkout there is never touched).
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -az --delete --exclude .git --exclude htmlcov --exclude '__pycache__' \
      --exclude '*.egg-info' --exclude '.pytest_cache' --exclude 'coverage.xml' \
      --exclude '.coverage' --exclude 'benchmarks/static_treecode/results' \
      ./ seitz1:research/CosmoFuse-static-treecode/
