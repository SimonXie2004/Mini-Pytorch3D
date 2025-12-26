#ifndef MINI_PYTORCH3D_RENDER_H
#define MINI_PYTORCH3D_RENDER_H

#include <mini_pytorch3d/math_utils.h>
#include <torch/types.h>


#include <torch/torch.h>


namespace mini_pytorch3d {

    torch::Tensor example();

    // full diffrential rasterization render function
    torch::Tensor render(const mesh& m, const camera& cam, const diffrast_args& args);
    
}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_RENDER_H