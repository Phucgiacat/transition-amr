#!/bin/bash
# =============================================================================
# setup.sh  -  Environment setup for IBM transition-amr-parser on Google Colab
# =============================================================================
# Run once to create the Python 3.8 virtual environment and install all deps.
#
# Usage (in a Colab cell):
#   !bash /content/transition-amr/setup.sh
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "  IBM Transition AMR Parser - Environment Setup"
echo "============================================================"

# 1) Navigate to the repo root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] Repo dir: ${REPO_DIR}"
cd "${REPO_DIR}"

# 2) Install uv (fast pip)
pip install -q uv

# 3) Create the .venv with Python 3.8
rm -rf .venv
uv venv .venv --python 3.8

PYTHON=".venv/bin/python"

# 4) Base build tools
uv pip install --python "${PYTHON}" pip setuptools wheel

# 5) Runtime dependencies
uv pip install --python "${PYTHON}" matplotlib ipdb progressbar2 penman

# 6) PyTorch stack (IBM README - cu117 build)
uv pip install --python "${PYTHON}" \
    --index-url https://download.pytorch.org/whl/cu117 \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1

# numpy 1.19.x required by fairseq 0.10.2 (uses np.float)
uv pip install --python "${PYTHON}" 'numpy>=1.19,<1.20'

# fairseq 0.10.2 provides __best_fitting_dtype used by the parser
uv pip install --python "${PYTHON}" fairseq==0.10.2

# 7) Install the parser package from the local source
uv pip install --python "${PYTHON}" -e .

# 8) Add src/ to site-packages path (.pth file)
SITE_PACKAGES=$(
  "${PYTHON}" -c \
    "import site, sysconfig
s = getattr(site, 'getsitepackages', None)
print(s()[0] if s and s() else sysconfig.get_paths()['purelib'])"
)
SRC_DIR="$(realpath src)"
PTH_FILE="${SITE_PACKAGES}/transition_amr_src.pth"
echo "${SRC_DIR}" > "${PTH_FILE}"
echo "[INFO] Added src path: ${PTH_FILE} -> ${SRC_DIR}"

# 9) Patch fairseq_ext __init__.py
FAIRSEQ_INIT="src/fairseq_ext/__init__.py"
cat > "${FAIRSEQ_INIT}" << 'PATCH_EOF'
"""Register fairseq extension modules.

In some editable-install environments, absolute imports like
    from fairseq_ext.data ...
can fail during early package initialization.
We proactively load the local fairseq_ext.data package from disk to keep
task auto-registration stable.
"""

import importlib
import importlib.util
from pathlib import Path
import sys


def _ensure_local_data_package() -> None:
    """Ensure fairseq_ext.data resolves to this repo's local package."""
    try:
        importlib.import_module("fairseq_ext.data")
        return
    except Exception:
        pass

    pkg_dir = Path(__file__).resolve().parent
    src_dir = pkg_dir.parent
    data_init = pkg_dir / "data" / "__init__.py"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if data_init.exists():
        spec = importlib.util.spec_from_file_location(
            "fairseq_ext.data",
            str(data_init),
            submodule_search_locations=[str(data_init.parent)],
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["fairseq_ext.data"] = module
            spec.loader.exec_module(module)


_ensure_local_data_package()

import fairseq_ext.criterions  # noqa: E402
import fairseq_ext.models      # noqa: E402
import fairseq_ext.tasks       # noqa: E402
PATCH_EOF
echo "[INFO] Patched ${FAIRSEQ_INIT}"

# 10) torch-scatter compatible with torch 1.13.1+cu117
uv pip install --python "${PYTHON}" --no-index torch-scatter \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# 11) Verify installation
echo ""
echo "[INFO] Verifying installation ..."
"${PYTHON}" - << 'VERIFY_EOF'
import torch, torch_scatter, fairseq, penman
import fairseq_ext.data
from transition_amr_parser.parse import AMRParser
print("  torch   :", torch.__version__)
print("  fairseq :", fairseq.__version__)
print("  cuda    :", torch.cuda.is_available())
print("  All imports OK!")
VERIFY_EOF

echo ""
echo "============================================================"
echo "  Setup complete.  Virtual env: .venv (Python 3.8)"
echo ""
echo "  Next step - run inference:"
echo "    bash inference.sh"
echo "  or:"
echo "    .venv/bin/python run.py \\"
echo "      --input  'train.en(2).jsonl' \\"
echo "      --output 'train-amr.txt' \\"
echo "      --process 'process.json'"
echo "============================================================"
