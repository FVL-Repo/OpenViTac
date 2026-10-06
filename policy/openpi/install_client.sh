#!/usr/bin/env bash
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-OpenViTac}"
OPENPI_CLIENT_PATH="${OPENPI_CLIENT_PATH:-}"

if [[ -f /etc/profile.d/conda.sh ]]; then
    source /etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Could not find conda. Activate the target environment manually or install openpi-client with pip."
    exit 2
fi

conda activate "${CONDA_ENV}"
python -m pip install -e "${OPENPI_CLIENT_PATH}"
