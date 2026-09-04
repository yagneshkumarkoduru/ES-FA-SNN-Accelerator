#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Project root: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_EXE}" -m pip install --upgrade pip
"${PYTHON_EXE}" -m pip install -r requirements.txt

echo "Environment setup complete."
