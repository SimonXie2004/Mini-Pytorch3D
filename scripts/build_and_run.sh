# export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX=$CUDA_HOME/bin/nvcc

rm -rf ./build
mkdir ./build && cd ./build
cmake ..
make

./src/mini_pytorch3d
