#!/bin/bash

set -euo pipefail



cd /content/transition-amr



rm -rf .venv

uv venv .venv --python 3.8

PYTHON=.venv/bin/python



uv pip install --python "${PYTHON}" pip setuptools wheel

uv pip install --python "${PYTHON}" matplotlib ipdb progressbar2 penman

uv pip install --python "${PYTHON}" --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1

uv pip install --python "${PYTHON}" 'numpy>=1.19,<1.20'

uv pip install --python "${PYTHON}" fairseq==0.10.2

uv pip install --python "${PYTHON}" -e .



SITE_PACKAGES=$("${PYTHON}" -c "import site,sysconfig; s=getattr(site,'getsitepackages',None); print(s()[0] if s and s() else sysconfig.get_paths()['purelib'])")

SRC_DIR=$(realpath src)

echo "${SRC_DIR}" > "${SITE_PACKAGES}/transition_amr_src.pth"



uv pip install --python "${PYTHON}" --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html



echo 'Verifying...'

"${PYTHON}" -c "import torch, torch_scatter, fairseq, penman; from transition_amr_parser.parse import AMRParser; print('torch', torch.__version__); print('fairseq', fairseq.__version__); print('ok')"

echo 'Setup done.'