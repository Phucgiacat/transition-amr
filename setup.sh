#!/bin/bash
# Setup script for AMR parser environment
# This script sets up the Python 3.8 environment with all dependencies

set -e  # Exit on error

echo "=========================================="
echo "Setting up AMR Parser Environment"
echo "=========================================="

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping setup..."
    echo "To reset, run: rm -rf .venv"
else
    echo ""
    echo "Creating Python 3.8 virtual environment..."
    uv venv .venv --python 3.8
    
    echo "Installing base packages..."
    uv pip install --python .venv pip setuptools wheel
    
    echo "Installing dependencies..."
    uv pip install --python .venv matplotlib ipdb progressbar2 penman
    
    echo "Installing PyTorch 1.13.1 with CUDA 11.7..."
    uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu117 \
        torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117 \
        torchaudio==0.13.1
    
    echo "Installing numpy (compatible with fairseq 0.10.2)..."
    uv pip install --python .venv 'numpy>=1.19,<1.20'
    
    echo "Installing fairseq 0.10.2..."
    uv pip install --python .venv fairseq==0.10.2
    
    echo "Installing transition_amr_parser from local source..."
    uv pip install --python .venv -e .
    
    echo "Installing torch-scatter..."
    uv pip install --python .venv --no-index torch-scatter \
        -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
    
    echo ""
    echo "Verifying installation..."
    .venv/bin/python -c "
import torch
import torch_scatter  
import fairseq
import penman
from transition_amr_parser.parse import AMRParser

print('✓ torch:', torch.__version__)
print('✓ fairseq:', fairseq.__version__)
print('✓ penman: ok')
print('✓ transition_amr_parser: ok')
print('✓ CUDA available:', torch.cuda.is_available())
"
    
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
fi
