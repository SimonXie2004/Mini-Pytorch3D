#!/usr/bin/env bash

set -e

# ====== change this if you need ======
export TORCH_VERSION="2.9.1"
export CUDA_VERSION="cu128"
# =====================================


# 1. download external/
TORCH_CONFIG="external/libtorch/share/cmake/Torch/TorchConfig.cmake"

if [ -f "$TORCH_CONFIG" ]; then
    echo "[libtorch] Found existing libtorch, skip download."
else
    mkdir -p external
    cd external

    LIBTORCH_URL="https://download.pytorch.org/libtorch/$CUDA_VERSION/libtorch-shared-with-deps-$TORCH_VERSION%2B$CUDA_VERSION.zip"
    LIBTORCH_ZIP="libtorch.zip"

    echo "[libtorch] libtorch not found, downloading with: "
    echo "Libtorch Version: $TORCH_VERSION + CUDA Version: $CUDA_VERSION"

    wget "$LIBTORCH_URL" -O "$LIBTORCH_ZIP"
    unzip -q "$LIBTORCH_ZIP"
    rm "$LIBTORCH_ZIP"

    echo "[libtorch] Downloaded to external/libtorch"
    cd ..
fi

mkdir -p external/stb
wget https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image_write.h -O external/stb/stb_image_write.h


# 2. download data/*
# stanford dragon: http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz
# stanford bunny: http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
# utah teapot: https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj

rm -rf data/downloads
mkdir -p data/downloads

cd data/downloads # <mini_pytorch3d_root>/data/downloads
wget http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz
wget http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
wget https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj
tar -xvf dragon_recon.tar.gz
tar -xvf bunny.tar.gz

cd ../.. # <mini_pytorch3d_root>/
python scripts/normalize_meshes.py --input_mesh data/downloads/dragon_recon/dragon_vrip.ply --output_mesh data/stanford_dragon.obj
python scripts/normalize_meshes.py --input_mesh data/downloads/bunny/reconstruction/bun_zipper.ply --output_mesh data/stanford_bunny.obj
python scripts/normalize_meshes.py --input_mesh data/downloads/teapot.obj --output_mesh data/utah_teapot.obj

echo "[data] Downloaded and normalized meshes are saved to data/*.obj"
