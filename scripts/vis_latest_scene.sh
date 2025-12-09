#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scene_number>"
  echo "Example: $0 300   # visualizes val-0300.bin"
  exit 1
fi

SCENE_NUM=$(printf "%04d" "$1")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_PATH="$REPO_ROOT/wandb/latest-run/files/scenes/val-${SCENE_NUM}.bin"

if [[ ! -f "$SCENE_PATH" ]]; then
  echo "Scene file not found: $SCENE_PATH"
  exit 2
fi

python "$REPO_ROOT/visual_utils/vis_terminal.py" -p "$SCENE_PATH"

