#ifndef MINI_PYTORCH3D_MATH_UTILS_H
#define MINI_PYTORCH3D_MATH_UTILS_H

#include <vector>

namespace mini_pytorch3d {
    struct float3 { float x, y, z; };
    struct float4 { float x, y, z, w; };
    
    struct mat3 { float x1, x2, x3;
                  float y1, y2, y3;
                  float z1, z2, z3; };
    struct mat4 { float x1, x2, x3, x4;
                  float y1, y2, y3, y4;
                  float z1, z2, z3, z4;
                         float w1, w2, w3, w4; };

    struct mesh {
        std::vector<float3> vertices;
        std::vector<float3> colors;
        std::vector<float3> faces;
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

    std::vector<float3> rand_range(std::size_t n);

}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_MATH_UTILS_H