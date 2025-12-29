#include <mini_pytorch3d/render.h>
#include <iostream>
#include <utility>

namespace mini_pytorch3d {
    
    // implemenation in cuda/diffrast.cu
    torch::Tensor rast(const triangle_buffer& tb, int image_width, int image_height, const torch::Tensor& light_dir);
    torch::Tensor sequential_diffrast(const triangle_buffer& tb, const diffrast_args& args, int image_width, int image_height, const torch::Tensor& light_dir);
    torch::Tensor parallel_diffrast(const triangle_buffer& tb, const diffrast_args& args, int image_width, int image_height, const torch::Tensor& light_dir);

    torch::Tensor camera_to_mvp_matrix(const camera& cam) {
        torch::Tensor M = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32));

        torch::Tensor V = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor eye = torch::from_blob(
            const_cast<float*>(cam.position.data()),
            {3},
            torch::kFloat
        ).clone();
        torch::Tensor look_at = torch::from_blob(
            const_cast<float*>(cam.look_at.data()),
            {3},
            torch::kFloat
        ).clone();
        torch::Tensor up = torch::from_blob(
            const_cast<float*>(cam.up.data()),
            {3},
            torch::kFloat
        ).clone();

        torch::Tensor f = (look_at - eye);
        f = f / torch::norm(f);
        torch::Tensor s = torch::cross(f, up, /*dim=*/0);
        s = s / torch::norm(s);
        torch::Tensor u = torch::cross(s, f, /*dim=*/0);

        V.index_put_({0, 0}, s[0].item());
        V.index_put_({0, 1}, s[1].item());
        V.index_put_({0, 2}, s[2].item());
        V.index_put_({1, 0}, u[0].item());
        V.index_put_({1, 1}, u[1].item());
        V.index_put_({1, 2}, u[2].item());
        V.index_put_({2, 0}, -f[0].item());
        V.index_put_({2, 1}, -f[1].item());
        V.index_put_({2, 2}, -f[2].item());
        V.index_put_({0, 3}, -torch::dot(s, eye).item());
        V.index_put_({1, 3}, -torch::dot(u, eye).item());
        V.index_put_({2, 3}, torch::dot(f, eye).item());

        float fov_rad = cam.fov * M_PI / 180.0f;
        float tan_half_fov = tan(fov_rad / 2.0f);
        float zn = cam.near_plane;
        float zf = cam.far_plane;

        torch::Tensor P = torch::zeros({4,4}, torch::TensorOptions().dtype(torch::kFloat32));
        P.index_put_({0,0}, 1.0f / (cam.aspect_ratio * tan_half_fov));
        P.index_put_({1,1}, 1.0f / (tan_half_fov));
        P.index_put_({2,2}, -(zf + zn) / (zf - zn));
        P.index_put_({2,3}, -(2.0f * zf * zn) / (zf - zn));
        P.index_put_({3,2}, -1.0f);

        torch::Tensor MVP = torch::matmul(P, torch::matmul(V, M));
        return MVP;
    }

    std::pair<mini_pytorch3d::triangle_buffer, torch::Tensor> transform_mesh_and_light(
        const mesh& m, const camera& cam) {
        auto MVP = camera_to_mvp_matrix(cam); // [4,4]
        auto MV = MVP.slice(0, 0, 3).slice(1, 0, 3); // [3,3]
        
        mini_pytorch3d::triangle_buffer tb {
            .colors = m.colors,
            .faces = m.faces - 1
        }; 

        // clip pos
        tb.clip_pos = torch::matmul(
            torch::cat({m.vertices, 
                torch::ones({static_cast<signed>(m.num_vertices), 1}, m.vertices.options())
            }, 1),
            MVP.transpose(0,1)
        ); // [N,4]

        // normal
        auto transformed_normals = torch::matmul(m.normals, MV.transpose(0, 1)); // [N,3]
        transformed_normals = transformed_normals / torch::norm(transformed_normals, 2, -1, true).clamp_min(1e-6f);
        tb.normals = transformed_normals; // [N,3]

        tb.inv_w = 1.0f / tb.clip_pos.index({"...", 3}).unsqueeze(-1); // [N,1]

        // light        
        torch::Tensor light_dir_world = torch::from_blob(
            const_cast<float*>(cam.position.data()),
            {3},
            torch::kFloat
        ).clone();
        auto light_dir_cam = torch::matmul(light_dir_world, MV.transpose(0, 1));
        light_dir_cam = light_dir_cam / torch::norm(light_dir_cam).clamp_min(1e-6f);

        return {tb, light_dir_cam};
    }

    torch::Tensor render(const mesh& m, const camera& cam, 
        const diffrast_args& args, int render_mode, bool requires_grad) {

        torch::Tensor image;

        m.vertices.set_requires_grad(requires_grad);
        m.colors.set_requires_grad(requires_grad);
        m.faces.set_requires_grad(false);

        auto [tb, light_dir_cam] = transform_mesh_and_light(m, cam); 
        
        if (render_mode == PARALLEL_DIFFRAST) {
            image = parallel_diffrast(tb, args, cam.W, cam.H, light_dir_cam);
        } 
        else if (render_mode == SEQUENTIAL_DIFFRAST) {
            image = sequential_diffrast(tb, args, cam.W, cam.H, light_dir_cam);
        } 
        else if (render_mode == SIMPLE_RAST) {
            image = rast(tb, cam.W, cam.H, light_dir_cam);
        }

        return image;
    }

}  // namespace mini_pytorch3d
