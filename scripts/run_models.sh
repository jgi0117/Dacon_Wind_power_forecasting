#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd $repo_root

python=.venv313/bin/python
[[ -x .venv313/Scripts/python.exe ]] && python=.venv313/Scripts/python.exe
[[ -x $python ]] || {
  echo 'Missing .venv313; run scripts/setup_env.sh first.' >&2
  exit 1
}

$python scripts/run_models.py $@
