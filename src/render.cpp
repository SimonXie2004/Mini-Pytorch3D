#include <mini_pytorch3d/render.h>

namespace mini_pytorch3d {
    
    // implemenation in cuda/diffrast.cu
    torch::Tensor rast(const triangle_buffer& tb, int image_width, int image_height);
    torch::Tensor diffrast(const triangle_buffer& tb, const diffrast_args& args, int image_width, int image_height);
    
    mini_pytorch3d::triangle_buffer mesh_to_triangle_buffer(const mesh& m, const camera& cam) {
        mini_pytorch3d::triangle_buffer tb {
            .colors = m.colors, 
            .faces = m.faces
        }; 

        auto MVP = camera_to_mvp_matrix(cam); // [4,4]

        tb.clip_pos = torch::matmul(
            torch::cat({m.vertices, 
                torch::ones({static_cast<signed>(m.num_vertices), 1}, m.vertices.options())
            }, 1),
            MVP.transpose(0,1)
        ); // [N,4]

        tb.inv_w = 1.0f / tb.clip_pos.index({"...", 3}).unsqueeze(-1); // [N,1]

        return tb;
    }

    torch::Tensor render(const mesh& m, const camera& cam, const diffrast_args& args = {}) {
        // track gradient for vertices and colors
        m.vertices.set_requires_grad(true);
        m.colors.set_requires_grad(true);
        m.faces.set_requires_grad(false);

        // render pipeline
        auto tb = mesh_to_triangle_buffer(m, cam);
        auto image = diffrast(tb, args, cam.W, cam.H); // [H,W,3]

        return image;
    }

}  // namespace mini_pytorch3d
