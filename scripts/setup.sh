#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${SUPERMIMIC_ROOT:-$ROOT/upstream}"
REV=28b15acb68ab3871f2ae1479964d53457ac066c6
if [[ ! -d "$UPSTREAM" ]]; then
  git clone https://github.com/Baojch/SuperMimic_Dyn.git "$UPSTREAM"
  git -C "$UPSTREAM" checkout --detach "$REV"
fi
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$REV" ]] || { echo 'Upstream revision mismatch; use a clean pinned checkout.' >&2; exit 1; }
python3.12 -m venv "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip install 'torch==2.14.1' 'warp-lang==1.17.0' 'mujoco==3.16.0' 'mujoco-warp==3.16.0' -e "$UPSTREAM/vendor/mjlab" -e "$UPSTREAM" -r "$ROOT/requirements.txt"
"$PY" -m pip check
printf 'Ready. Python: %s\n' "$PY"
