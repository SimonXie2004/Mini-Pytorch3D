# export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# cmake -S <source_dir> -B <build_dir> [options...]
# cmake -S . -B build # release mode
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g -rdynamic" \
  -DCMAKE_EXE_LINKER_FLAGS="-rdynamic" \
  -DCMAKE_CUDA_FLAGS_DEBUG="-O0 -g -G" # debug mode

# cmake --build <build_dir> [--target <tgt>] [--config <cfg>] [-- -jN]
cmake --build build -j$(nproc)

./build/src/mini_pytorch3d data/utah_teapot.obj output/output.png
