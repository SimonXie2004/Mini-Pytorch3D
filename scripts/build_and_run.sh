# export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cmake -S . -B build # cmake -S <source_dir> -B <build_dir> [options...]
cmake --build build -j$(nproc) # cmake --build <build_dir> [--target <tgt>] [--config <cfg>] [-- -jN]

nvcc -V

./build/src/mini_pytorch3d data/stanford_dragon.obj output/output.png
