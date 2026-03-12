#!/bin/bash
# Compatibility wrapper for users expecting set_environment.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/setup.sh"