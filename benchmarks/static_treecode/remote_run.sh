#!/usr/bin/env bash
# Usage: remote_run.sh <command...>   (runs on an A100 node of el9's usm-cluster partition,
# usm-cl-seitz1/2, inside the tree deployed by deploy_el9.sh, venv ~/.venvs/cosmo, with the
# deployed src/ shadowing the editable install). <command> is shell code, as before.
# SRUN_OPTS overrides the allocation, e.g. SRUN_OPTS="--gres=gpu:a100:1 -c 32 --mem=128G -t 2:00:00 -w usm-cl-seitz2"
set -euo pipefail
REMOTE=/project/ocean1-seitz/users/dgebauer/research/CosmoFuse-static-treecode
SRUN_OPTS=${SRUN_OPTS:-"--gres=gpu:a100:1 -c 16 --mem=64G -t 1:00:00"}
inner="source ~/.venvs/cosmo/bin/activate && cd $REMOTE && export PYTHONPATH=\$PWD/src && $*"
ssh -o BatchMode=yes el9-login "srun -p usm-cluster $SRUN_OPTS bash -c $(printf '%q' "$inner")"
