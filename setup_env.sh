#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Project directory: ${PROJECT_DIR}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required to install uv." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv was installed but is not on PATH." >&2
  echo "Add \$HOME/.local/bin to your shell PATH and rerun this script." >&2
  exit 1
fi

echo "Creating virtual environment with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

echo "Installing dependencies from requirements.txt..."
uv pip install --python "${VENV_DIR}/bin/python" -r "${PROJECT_DIR}/requirements.txt"

echo "Registering Jupyter kernel..."
uv run --python "${VENV_DIR}/bin/python" python -m ipykernel install \
  --user \
  --name bo-challenge-wk8 \
  --display-name "Python (bo-challenge-wk8)"

cat <<EOF

Environment setup complete.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Launch Jupyter with:
  jupyter lab

Or run the notebook kernel named:
  Python (bo-challenge-wk8)
EOF
