#include <torch/torch.h>
#include <mini_pytorch3d/types.h>
#include <iostream>

namespace mini_pytorch3d {

    

    inline float edge_fn(const vec2& a, const vec2& b, float px, float py) {
        return (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0]);
    }

    // Top-left rule
    inline bool is_top_left(const vec2& a, const vec2& b) {
        return (a[1] < b[1]) || (a[1] == b[1] && a[0] > b[0]);
    }

    // Implementation of parallel differentiable rasterization
    torch::Tensor parallel_diffrast(const triangle_buffer& tb, const diffrast_args& args, 
                            int image_width, int image_height, const torch::Tensor& light_dir) {
        
        
        return torch::zeros({image_height, image_width, 3}, torch::kU8); // Placeholder
    }

    // Implementation of sequential differentiable rasterization
    // (unrelated, just for performance comparison)
    torch::Tensor sequential_diffrast(const triangle_buffer& tb, const diffrast_args& args, 
                            int image_width, int image_height, const torch::Tensor& light_dir) {
        // r, g, b, running_max, weight_buffer
        auto image_buffer = torch::zeros({image_height, image_width, 5}, torch::kFloat32);
        image_buffer.index_put_({"...", 3}, 1e10f); 

        return image_buffer.slice(2, 0, 3); // Placeholder
    }

    // Implementation of NON-differentiable, SINGLE-thread rasterization 
    // (unrelated, just for fun)
    torch::Tensor rast(const triangle_buffer& tb, int image_width, int image_height, const torch::Tensor& light_dir) {
        // r, g, b, z
        auto image_buffer = torch::zeros({image_height, image_width, 4}, torch::kFloat32);
        image_buffer.index_put_({"...", 3}, 1e10f); 

        for (std::size_t i = 0; i < tb.faces.size(0); ++i) {
            auto face = tb.faces[i];
            auto idx0 = face[0].item<int64_t>();
            auto idx1 = face[1].item<int64_t>();
            auto idx2 = face[2].item<int64_t>();
            
            auto v0 = tb.clip_pos[idx0];
            auto v1 = tb.clip_pos[idx1];
            auto v2 = tb.clip_pos[idx2];

            v0 = v0 / v0[3];
            v1 = v1 / v1[3];
            v2 = v2 / v2[3];

            // bbox rast
            vec2 sv0 = {
                (v0[0].item<float>() * 0.5f + 0.5f) * image_width,
                (-v0[1].item<float>() * 0.5f + 0.5f) * image_height
            };
            vec2 sv1 = {
                (v1[0].item<float>() * 0.5f + 0.5f) * image_width,
                (-v1[1].item<float>() * 0.5f + 0.5f) * image_height
            };
            vec2 sv2 = {
                (v2[0].item<float>() * 0.5f + 0.5f) * image_width,
                (-v2[1].item<float>() * 0.5f + 0.5f) * image_height
            };

            int xmin = std::max(0, (int)std::floor(std::min({sv0[0], sv1[0], sv2[0]})));
            int xmax = std::min(image_width - 1, (int)std::ceil(std::max({sv0[0], sv1[0], sv2[0]})));
            int ymin = std::max(0, (int)std::floor(std::min({sv0[1], sv1[1], sv2[1]})));
            int ymax = std::min(image_height - 1, (int)std::ceil(std::max({sv0[1], sv1[1], sv2[1]})));

            float area = edge_fn(sv0, sv1, sv2[0], sv2[1]);
            if (std::abs(area) < 1e-8f) continue;
            
            // rast loop
            for (int y = ymin; y <= ymax; ++y) {
                for (int x = xmin; x <= xmax; ++x) {

                    float px = x + 0.5f;
                    float py = y + 0.5f;

                    float e0 = edge_fn(sv1, sv2, px, py);
                    float e1 = edge_fn(sv2, sv0, px, py);
                    float e2 = edge_fn(sv0, sv1, px, py);

                    bool inside =
                        (e0 > 0 || (e0 == 0 && is_top_left(sv1, sv2))) &&
                        (e1 > 0 || (e1 == 0 && is_top_left(sv2, sv0))) &&
                        (e2 > 0 || (e2 == 0 && is_top_left(sv0, sv1)));
                    if (!inside) continue;

                    float w0 = e0 / area;
                    float w1 = e1 / area;
                    float w2 = 1.0f - w0 - w1;
                    if (w0 < 0 || w1 < 0 || w2 < 0) {
                        continue;
                    }

                    float depth =
                        w0 * v0[2].item<float>() +
                        w1 * v1[2].item<float>() +
                        w2 * v2[2].item<float>();

                    if (depth < image_buffer[y][x][3].item<float>()) {
                        // phong shading (diffuse only)
                        auto normal = w0 * tb.normals[idx0] + 
                                     w1 * tb.normals[idx1] + 
                                     w2 * tb.normals[idx2];
                        
                        auto normal_normalized = normal / torch::norm(normal);
                        // Use passed-in light direction
                        float n_dot_l_1 = torch::dot(normal_normalized, light_dir).item<float>();
                        float n_dot_l_2 = torch::dot(-normal_normalized, light_dir).item<float>();
                        float diffuse = std::max(0.0f, std::max(n_dot_l_1, n_dot_l_2)); // in case the normal points the opposite
                        
                        auto base_color = w0 * tb.colors[idx0] + 
                                         w1 * tb.colors[idx1] + 
                                         w2 * tb.colors[idx2];
                        
                        auto color = base_color * diffuse;
                        
                        image_buffer[y][x][0] = color[0].item<float>();
                        image_buffer[y][x][1] = color[1].item<float>();
                        image_buffer[y][x][2] = color[2].item<float>();
                        image_buffer[y][x][3] = depth;
                    }
                }
            }
        }

        return image_buffer.slice(2, 0, 3);
    }

} // namespace mini_pytorch3d
