cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

project(mini_pytorch3d LANGUAGES CXX CUDA)

set(Torch_DIR "external/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_subdirectory(src)

set_property(TARGET mini_pytorch3d PROPERTY CUDA_SEPARABLE_COMPILATION ON)
