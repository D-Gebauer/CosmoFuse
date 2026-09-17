#!/usr/bin/env bash
# Usage: remote_run.sh <command...>   (runs on seitz1 inside the deployed tree,
# conda env cosmo, with the deployed src/ shadowing the editable install)
set -euo pipefail
ssh -o BatchMode=yes seitz1 "source ~/anaconda3/etc/profile.d/conda.sh && conda activate cosmo && cd ~/research/CosmoFuse-static-treecode && export PYTHONPATH=\$PWD/src && $*"
