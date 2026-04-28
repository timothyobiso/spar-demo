#!/usr/bin/env bash
# Launch the steering demo on a remote GPU box (Vast.ai etc).
# Run this on the box, from the project root.
#
# Usage:
#   bash run_remote.sh                 # local-only (use SSH tunnel from laptop)
#   STEER_SHARE=1 bash run_remote.sh   # public *.gradio.live link
#
# Env vars (optional):
#   STEER_MODEL    HF model id (default Qwen/Qwen3-4B)
#   STEER_PROBES   path to probes.pkl (default ./results/qwen3.5-4b/probes.pkl)
#   STEER_LAYER    target layer index (default highest)
#   STEER_PORT     server port (default 7860)
#   STEER_SHARE    set to 1 for Gradio share link
#   HF_TOKEN       HuggingFace token if model needs gated access

set -euo pipefail

# Vast PyTorch images already have torch+cuda. pip is a no-op for satisfied deps.
# Drop the `spaces` package — it's HF-Spaces-only and unused here.
REQ_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP"' EXIT
grep -v '^spaces' requirements.txt > "$REQ_TMP"

PIP_FLAGS=(--break-system-packages)
python3 -m pip install "${PIP_FLAGS[@]}" -r "$REQ_TMP"
# requirements.txt pins huggingface_hub<1.0 (for HF Spaces gradio 5.4 compat).
# On Vast we need transformers>=5 to support newer Qwen models — upgrade explicitly,
# which pulls huggingface_hub>=1.0, and pair with a gradio version that supports it.
python3 -m pip install "${PIP_FLAGS[@]}" -U transformers huggingface_hub
python3 -m pip install "${PIP_FLAGS[@]}" -U gradio

PROBES_PATH="${STEER_PROBES:-./results/qwen3.5-4b/probes.pkl}"
if [ ! -f "$PROBES_PATH" ]; then
    echo "ERROR: probes.pkl not found at $PROBES_PATH"
    echo "From your laptop, scp/rsync it up, e.g.:"
    echo "  bash push_to_vast.sh <ssh_host> <ssh_port>"
    exit 1
fi

export STEER_PORT="${STEER_PORT:-7860}"

echo
echo "Launching app.py on 0.0.0.0:${STEER_PORT}"
if [ "${STEER_SHARE:-0}" = "1" ]; then
    echo "  share=True → public *.gradio.live URL will print below"
else
    echo "  local-only → from laptop:  ssh -L ${STEER_PORT}:localhost:${STEER_PORT} -p <ssh_port> <user@host>"
    echo "                              then open http://localhost:${STEER_PORT}"
fi
echo

exec python3 app.py
