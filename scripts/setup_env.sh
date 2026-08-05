#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd $repo_root

export UV_CACHE_DIR=$repo_root/.uv-cache
export UV_PYTHON_INSTALL_DIR=$repo_root/.uv-python

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv was not found. Install uv and run this script again.' >&2
  exit 1
fi

[[ -x .venv313/Scripts/python.exe || -x .venv313/bin/python ]] || uv venv .venv313 --python 3.13

python=.venv313/bin/python
[[ -x .venv313/Scripts/python.exe ]] && python=.venv313/Scripts/python.exe

uv pip install --python $python -r requirements-py313.txt
uv pip install --python $python --editable .

$python --version
