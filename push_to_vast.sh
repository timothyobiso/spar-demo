#!/usr/bin/env bash
# Rsync the spar_demo project (code + probes.pkl) up to a Vast.ai box.
# Run from your laptop, in the project root.
#
# Usage: bash push_to_vast.sh <ssh_host> <ssh_port> [remote_dir]
#   e.g. bash push_to_vast.sh root@ssh4.vast.ai 12345
#        bash push_to_vast.sh root@ssh4.vast.ai 12345 /workspace/spar_demo

set -euo pipefail

HOST="${1:?usage: push_to_vast.sh <ssh_host> <ssh_port> [remote_dir]}"
PORT="${2:?usage: push_to_vast.sh <ssh_host> <ssh_port> [remote_dir]}"
REMOTE_DIR="${3:-spar_demo}"

rsync -avz --progress \
    -e "ssh -p $PORT" \
    --exclude __pycache__ \
    --exclude .venv \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    ./ "$HOST:$REMOTE_DIR/"

cat <<EOF

Uploaded to $HOST:$REMOTE_DIR

Next steps:
  1. SSH in and launch:
       ssh -p $PORT $HOST
       cd $REMOTE_DIR && bash run_remote.sh

  2. In a second laptop terminal, tunnel the port:
       ssh -L 7860:localhost:7860 -p $PORT $HOST
     then open http://localhost:7860 in your browser.

  Or, if tunneling is blocked, on the box run:
       STEER_SHARE=1 bash run_remote.sh
     and use the *.gradio.live URL it prints.
EOF
