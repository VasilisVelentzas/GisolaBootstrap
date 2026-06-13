#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating conda environment..."
conda env create -f "$REPO_DIR/environment.yml" || true

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gisola2

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
echo "   conda activate gisola2"
echo "   cd ~/GisolaBootstrap/src"
echo "   ./gisola2Bootstrap.py -c ../test/config.yaml --event-xml ../test/benchmark/event.xml"
echo ""
echo "============================================"
echo " Optional: GPU support (NVIDIA CUDA)"
echo "============================================"
echo ""
echo " To enable GPU acceleration (requires NVIDIA GPU + CUDA driver):"
echo ""
echo " 1. Add the NVIDIA CUDA apt repository (WSL2 / Ubuntu 24.04):"
echo "      wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
echo "      sudo dpkg -i cuda-keyring_1.1-1_all.deb"
echo "      sudo apt-get update"
echo ""
echo " 2. Install required CUDA libraries:"
echo "      sudo apt-get install -y libnvvm4 cuda-cudart-12-9 cuda-nvcc-12-9"
echo "      sudo ln -sf /usr/lib/x86_64-linux-gnu/libnvvm.so.4 /usr/lib/x86_64-linux-gnu/libnvvm.so"
echo ""
echo " 3. Make the libraries available to numba on every conda activation:"
echo "      mkdir -p \$(conda info --base)/envs/gisola2/etc/conda/activate.d"
echo "      cat > \$(conda info --base)/envs/gisola2/etc/conda/activate.d/cuda.sh << 'EOF'"
echo "export CUDA_HOME=/usr/local/cuda-12.9"
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH"
echo "EOF"
echo ""
echo " 4. Set ComputeDevice: GPU in your config.yaml"
echo ""
