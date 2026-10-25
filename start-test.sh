#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /usr/local/bin/spark-heavy-run \
  /home/baconnier/ComfyUI-qwen21/.venv/bin/python main.py \
  --listen 127.0.0.1 --port 8192 --reserve-vram 24 \
  --disable-pinned-memory --disable-all-custom-nodes --disable-auto-launch
