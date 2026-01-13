#include <mini_pytorch3d/render.h>
#include <mini_pytorch3d/constants.h>
#include <mini_pytorch3d/legacy.h>

#include <iostream>
#include <utility>

#include <cuda.h>
#include <c10/cuda/CUDAGuard.h> // get stream

namespace mini_pytorch3d {

    // implemenation in src/legacy.cpp (for comparison)
    torch::Tensor rast(const triangle_buffer& tb, int image_width, int image_height, const torch::Tensor& light_dir);
    torch::Tensor sequential_diffrast(const triangle_buffer& tb, const diffrast_args& args, int image_width, int image_height, const torch::Tensor& light_dir);
    // implemenation in src/render.cpp (here, below)
    torch::Tensor parallel_diffrast(const triangle_buffer& tb, const diffrast_args& args, int image_width, int image_height, const torch::Tensor& light_dir);
    // implemenation in cuda/diffrast.cu (NOT here)
    void launch_parallel_diffrast_kernel(
        const float* clip_pos,
        const float* inv_w,
        const float* colors,
        const float* normals,
        const int64_t* faces,
        const int* tile_offsets,
        const int* tile_faces,
        int W, int H, float sigma, float gamma, 
        const float* light_dir,
        float* out
    );
    // implemenation in cuda/binning.cu (NOT here)
    void launch_binning_two_pass(
        const float* clip_pos,
        const float* inv_w,
        const int64_t* faces,
        int F,
        int W, int H,
        int* tile_offsets,
        int* tile_faces_flat,
        int* total_pairs_out,
        cudaStream_t stream
    );
    // implementation in cuda/backward.cu (NOT here)

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
            .faces = (m.faces - 1).to(torch::kInt64),
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

    // Main render function; for manager use
    torch::Tensor render(const mesh& m, const camera& cam, 
        const diffrast_args& args, int render_mode, bool requires_grad) {

        torch::Tensor image;

        m.vertices.set_requires_grad(requires_grad);
        m.colors.set_requires_grad(requires_grad);
        m.faces.set_requires_grad(false);
        m.normals.set_requires_grad(false);

        auto [tb, light_dir_cam] = transform_mesh_and_light(m, cam); 
        
        auto start = std::chrono::high_resolution_clock::now();

        if (render_mode == PARALLEL_DIFFRAST) {
            tb.clip_pos = tb.clip_pos.to(torch::kCUDA);
            tb.inv_w    = tb.inv_w.to(torch::kCUDA);
            tb.colors   = tb.colors.to(torch::kCUDA);
            tb.normals  = tb.normals.to(torch::kCUDA);
            tb.faces    = tb.faces.to(torch::kCUDA);
            image = parallel_diffrast(tb, args, cam.W, cam.H, light_dir_cam);
        } 
        else if (render_mode == SEQUENTIAL_DIFFRAST) {
            image = sequential_diffrast(tb, args, cam.W, cam.H, light_dir_cam);
        } 
        else if (render_mode == SIMPLE_RAST) {
            image = rast(tb, cam.W, cam.H, light_dir_cam);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Rendering took " << duration.count() << " ms" << std::endl;

        return image;
    }

    // Implementation of parallel differentiable rasterization
    torch::Tensor parallel_diffrast(
        const triangle_buffer& tb,
        const diffrast_args& args,
        int image_width,
        int image_height,
        const torch::Tensor& light_dir
    ) {
        TORCH_CHECK(tb.clip_pos.is_cuda(), "triangle_buffer.clip_pos must be on CUDA");
        at::cuda::CUDAGuard guard(tb.clip_pos.device());
        cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

        auto clip   = tb.clip_pos.contiguous();
        auto invw   = tb.inv_w.view({-1}).contiguous();
        auto colors = tb.colors.contiguous();
        auto normals= tb.normals.contiguous();
        auto faces  = tb.faces.to(torch::kInt64).contiguous();
        auto light  = light_dir.to(torch::kCUDA).to(torch::kFloat32).contiguous();

        int F = (int)faces.size(0);

        int tiles_x = (image_width  + TILE - 1) / TILE;
        int tiles_y = (image_height + TILE - 1) / TILE;
        int num_tiles = tiles_x * tiles_y;

        // allocate binning output on cuda
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto tile_offsets = torch::empty({num_tiles + 1}, opts_i32);
        auto total_pairs_dev = torch::empty({1}, opts_i32);
        int64_t cap = std::max((int64_t)F * MAX_TILES_PER_TRIANGLE, 1l);
        auto tile_faces_flat_cap = torch::empty({cap}, opts_i32);

        launch_binning_two_pass(
            clip.data_ptr<float>(),
            invw.data_ptr<float>(),
            faces.data_ptr<int64_t>(),
            F,
            image_width,
            image_height,
            tile_offsets.data_ptr<int>(),
            tile_faces_flat_cap.data_ptr<int>(),
            total_pairs_dev.data_ptr<int>(),
            stream
        );

        // read total_pairs (small scalar copy; stays async-safe wrt stream ordering)
        int total_pairs = total_pairs_dev.cpu().item<int>();
        TORCH_CHECK(total_pairs >= 0, "total_pairs < 0, binning failed?");
        TORCH_CHECK((int64_t)total_pairs <= cap,
                    "tile_faces_flat capacity too small: total_pairs=", total_pairs,
                    " cap=", cap,
                    " (increase cap multiplier, e.g., F*16)");
        auto tile_faces_flat = tile_faces_flat_cap.narrow(0, 0, total_pairs);

        // output & rast kernel
        auto out = torch::zeros(
            {image_height, image_width, 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
        );

        launch_parallel_diffrast_kernel(
            clip.data_ptr<float>(),
            invw.data_ptr<float>(),
            colors.data_ptr<float>(),
            normals.data_ptr<float>(),
            faces.data_ptr<int64_t>(),
            tile_offsets.data_ptr<int>(),
            tile_faces_flat.data_ptr<int>(),
            image_width,
            image_height,
            args.sigma, args.gamma, 
            light.data_ptr<float>(),
            out.data_ptr<float>()
        );

        return out;
    }

    
}  // namespace mini_pytorch3d
