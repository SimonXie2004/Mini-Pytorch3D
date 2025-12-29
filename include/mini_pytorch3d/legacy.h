#ifndef MINI_PYTORCH3D_LEGACY_H
#define MINI_PYTORCH3D_LEGACY_H

#include <mini_pytorch3d/types.h>
#include <torch/types.h>

namespace mini_pytorch3d {

    torch::Tensor sequential_diffrast(const triangle_buffer& tb, const diffrast_args& args, 
                            int image_width, int image_height, const torch::Tensor& light_dir); 

    torch::Tensor rast(const triangle_buffer& tb, 
        int image_width, int image_height, const torch::Tensor& light_dir);

}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_LEGACY_H