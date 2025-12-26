#include <mini_pytorch3d/render.h>

namespace mini_pytorch3d {
    
    // implemenation in cuda/diffrast.cu
    torch::Tensor diffrast(const triangle_buffer& tb, const diffrast_args& args); 

    mini_pytorch3d::triangle_buffer mesh_to_triangle_buffer(const mesh& m, const camera& cam) {
        mini_pytorch3d::triangle_buffer tb;

        return tb;
    }

    torch::Tensor render(const mesh& m, const camera& cam, const diffrast_args& args = {}) {
        // track gradient for vertices and colors
        m.vertices.set_requires_grad(true);
        m.colors.set_requires_grad(true);
        m.faces.set_requires_grad(false);

        // render pipeline
        // 1. convert to triangle buffers
        auto tb = mesh_to_triangle_buffer(m, cam);

        // 2. rasterization with differentiable renderer
        auto image = diffrast(tb, args);

        return image;
    }

}  // namespace mini_pytorch3d
