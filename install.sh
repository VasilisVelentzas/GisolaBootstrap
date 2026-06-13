#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating conda environment..."
conda env create -f "$REPO_DIR/environment.yml" || true

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gisola

echo "Compiling Axitra..."
cd "$REPO_DIR/src/core/green"
make

echo "Setting executable permissions..."
chmod +x "$REPO_DIR/src/gisolaBootstrap.py"

echo "Pre-compiling Numba kernels (one-time, ~10s)..."
cd "$REPO_DIR/src"
python precompile.py

echo "Adding ulimit to ~/.bashrc..."
if ! grep -q "ulimit -s unlimited" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Gisola: fix segmentation fault with large stack" >> ~/.bashrc
    echo "ulimit -s unlimited" >> ~/.bashrc
    echo "  -> Added 'ulimit -s unlimited' to ~/.bashrc"
else
    echo "  -> Already present in ~/.bashrc, skipping"
fi

echo ""
echo "============================================"
echo " Installation completed."
echo "============================================"
echo ""
echo " To start using Gisola, open a new terminal"
echo " (so ulimit takes effect) and run:"
echo ""
echo "   conda activate gisola"
echo "   cd ~/Gisola2/src"
echo "   ./gisola.py -c ../test/config.yaml --event-xml ../test/benchmark/event.xml"
echo ""
