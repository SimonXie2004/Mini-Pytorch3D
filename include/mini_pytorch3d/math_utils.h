#ifndef MINI_PYTORCH3D_MATH_UTILS_H
#define MINI_PYTORCH3D_MATH_UTILS_H

#include <array>
#include <torch/types.h>

namespace mini_pytorch3d {
    using float3 = std::array<float, 3>;
    using float4 = std::array<float, 4>;

    using mat3 = std::array<float3, 3>;
    using mat4 = std::array<float4, 4>;
    
    /*
    Must satisfy:
        static_assert(std::is_standard_layout_v<float3>);
        static_assert(std::is_trivially_copyable_v<float3>);
        static_assert(sizeof(float3) == 3 * sizeof(float));
    (Also same for other dtypes)

    Maybe i should deprecate `using float3 = struct float3 { float x, y, z; };`
        which is memory unsafe
    */

    struct mesh {
        torch::Tensor vertices;
        torch::Tensor colors;
        torch::Tensor faces;
        std::size_t num_vertices;
        std::size_t num_faces;
    };

    struct camera {
        float3 position;
        float3 look_at;
        float3 up;
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
        torch::Tensor faces;    // [N, 3]
    };

    torch::Tensor camera_to_mvp_matrix(const camera& cam);

}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_MATH_UTILS_H