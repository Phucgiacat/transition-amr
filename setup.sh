#!/bin/bash

set -euo pipefail



cd /content/transition-amr-parser



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



# Patch fairseq_ext package init to preload local data package

cat > src/fairseq_ext/__init__.py << 'PATCH_EOF'

"""Register fairseq extension modules.



In some editable-install environments, absolute imports like

from fairseq_ext.data ... can fail during early package initialization.

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



uv pip install --python "${PYTHON}" --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html



echo 'Verifying...'

"${PYTHON}" -c "import torch, torch_scatter, fairseq, penman; import fairseq_ext.data; from transition_amr_parser.parse import AMRParser; print('torch', torch.__version__); print('fairseq', fairseq.__version__); print('cuda', torch.cuda.is_available()); print('ok')"

echo 'Setup done.'