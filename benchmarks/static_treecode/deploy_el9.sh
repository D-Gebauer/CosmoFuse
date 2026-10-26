#!/usr/bin/env bash
# Sync the working tree to a separate directory on the el9 cluster's shared storage (the user's own
# research/CosmoFuse checkout there is never touched). Same directory seitz1 used as ~/research/...;
# run things on the A100 nodes with remote_run.sh. The P filter keeps probe JSONs that scripts wrote next to
# themselves on the cluster (e.g. aperture_reuse_probe_512.json) from being --delete'd.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -az --delete --exclude .git --exclude htmlcov --exclude '__pycache__' \
      --exclude '*.egg-info' --exclude '.pytest_cache' --exclude 'coverage.xml' \
      --exclude '.coverage' --exclude 'benchmarks/static_treecode/results' \
      --filter 'P benchmarks/static_treecode/*.json' \
      ./ el9-login:/project/ocean1-seitz/users/dgebauer/research/CosmoFuse-static-treecode/
