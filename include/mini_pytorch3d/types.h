#ifndef MINI_PYTORCH3D_TYPES_H
#define MINI_PYTORCH3D_TYPES_H

#include <array>
#include <torch/types.h>

namespace mini_pytorch3d {
    enum render_options {
        PARALLEL_DIFFRAST = 0, 
        SEQUENTIAL_DIFFRAST = 1, 
        SIMPLE_RAST = 2
    }; 

    enum grad_options {
        REQUIRES_GRAD = 0, 
        NO_GRAD = 1
    }; 

    using vec2 = std::array<float, 2>;
    using vec3 = std::array<float, 3>;

    /*
    Must satisfy:
        static_assert(std::is_standard_layout_v<vec3>);
        static_assert(std::is_trivially_copyable_v<vec3>);
        static_assert(sizeof(vec3) == 3 * sizeof(float));
    (Also same for other dtypes)

    Maybe i should deprecate `using vec3 = struct vec3 { float x, y, z; };`
        which is memory unsafe due to padding?
    */

    struct mesh {
        torch::Tensor vertices;  // [N, 3]
        torch::Tensor colors;    // [N, 3]
        torch::Tensor normals;   // [N, 3]
        torch::Tensor faces;     // [M, 3]
        std::size_t num_vertices;
        std::size_t num_faces;
    };

    struct camera {
        vec3 position;
        vec3 look_at;
        vec3 up;
        float fov; // in degrees
        float aspect_ratio;
        float near_plane;
        float far_plane;
        std::size_t H, W;
    };

    struct diffrast_args {
        float sigma;
        float gamma; 
    };

    struct triangle_buffer {
        torch::Tensor clip_pos; // [N, 4]
        torch::Tensor inv_w;    // [N, 1]
        torch::Tensor colors;   // [N, 3]
        torch::Tensor normals;  // [N, 3]
        torch::Tensor faces;    // [N, 3]
    };

}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_TYPES_H