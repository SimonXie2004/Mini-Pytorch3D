#ifndef MINI_PYTORCH3D_RENDER_H
#define MINI_PYTORCH3D_RENDER_H

#include <mini_pytorch3d/types.h>
#include <torch/types.h>

namespace mini_pytorch3d {

    // full diffrential rasterization render function
    torch::Tensor render(const mesh& m, const camera& cam, 
        const diffrast_args& args = {}, int render_mode = PARALLEL_DIFFRAST, bool requires_grad = false);
    
}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_RENDER_H