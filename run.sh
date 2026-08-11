#!/usr/bin/env bash
# Launch the batch image editor. Designed to be run inside tmux on a vast.ai
# instance so it survives SSH disconnects:
#
#   tmux new -s editor
#   ./run.sh
#   # detach with Ctrl-b then d ; reattach later with: tmux attach -t editor
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="${VENV:-.venv}"

# Activate a virtualenv if one is present. vast.ai images often provide a
# preinstalled torch in the system python, in which case skip the venv entirely
# by setting VENV=none.
if [[ "${VENV}" != "none" && -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  echo "Using virtualenv: ${VENV}"
else
  echo "Using system python: $(command -v python3)"
fi

PYTHON="${PYTHON:-python3}"

# Keep model downloads on the big disk rather than the small root volume.
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
mkdir -p "${HF_HOME}"

# Faster, more resilient downloads for large checkpoints.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Gradio phones home for usage analytics by default; pointless on a rented box.
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

PORT="${IMGBATCH_PORT:-7860}"
echo "----------------------------------------------------------------"
echo " Batch Image Editor"
echo "   port       : ${PORT}"
echo "   HF cache   : ${HF_HOME}"
echo "   config     : ${IMGBATCH_CONFIG:-config.yaml}"
echo "----------------------------------------------------------------"

exec "${PYTHON}" app.py "$@"
