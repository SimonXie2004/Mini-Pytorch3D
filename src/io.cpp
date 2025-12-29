#include <mini_pytorch3d/io.h>
#include <mini_pytorch3d/types.h>
#include <torch/torch.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace mini_pytorch3d {

    struct mesh parse_obj(const std::string& file_path) {
        // only reads v lines and f lines; Ignores vt, vn, # lines.
        FILE *file = fopen(file_path.c_str(), "r");

        if (file == NULL) {
            perror("Failed to open file");
            return {};
        }

        std::vector<vec3> vertices;
        std::vector<vec3> faces;  

        char line[128];
        while (fgets(line, sizeof(line), file)) {
            if (line[0] == 'v') {
                vec3 vertex;
                sscanf(line, "v %f %f %f", &vertex[0], &vertex[1], &vertex[2]);
                vertices.push_back(vertex);
            } else if (line[0] == 'f') {
                vec3 face;
                sscanf(line, "f %f %f %f", &face[0], &face[1], &face[2]);
                faces.push_back(face);
            }
        }

        fclose(file);

        auto vertices_tensor = torch::from_blob(
            vertices.data(),
            {static_cast<long>(vertices.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone();
        
        auto faces_tensor = torch::from_blob(
            faces.data(),
            {static_cast<long>(faces.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone();

        // simple vertex_normal = avg(face_normal)
        auto normals_tensor = torch::zeros_like(vertices_tensor);
        
        for (size_t i = 0; i < faces.size(); ++i) {
            int idx0 = static_cast<int>(faces[i][0]) - 1;
            int idx1 = static_cast<int>(faces[i][1]) - 1;
            int idx2 = static_cast<int>(faces[i][2]) - 1;
            
            auto v0 = vertices_tensor[idx0];
            auto v1 = vertices_tensor[idx1];
            auto v2 = vertices_tensor[idx2];
            
            auto edge1 = v1 - v0;
            auto edge2 = v2 - v0;
            auto face_normal = torch::cross(edge1, edge2, /*dim=*/0);
            
            normals_tensor[idx0] += face_normal;
            normals_tensor[idx1] += face_normal;
            normals_tensor[idx2] += face_normal;
        }
        
        normals_tensor = normals_tensor / torch::norm(normals_tensor, 2, -1, true).clamp_min(1e-6f);

        // vertex color
        auto vertex_abs_max = torch::abs(vertices_tensor).max().item<float>();
        auto normalized_vertices = vertices_tensor / vertex_abs_max;
        auto vertex_colors = normalized_vertices * 0.5f + 0.5f;

        struct mesh m {
            .vertices = vertices_tensor,
            .colors = vertex_colors,
            .normals = normals_tensor,
            .faces = faces_tensor,
            .num_vertices = vertices.size(),
            .num_faces = faces.size()
        };
        return m;
    }

    void save_tensor_as_png(const torch::Tensor& img, const std::string& path) {
        TORCH_CHECK(img.dim() == 3 && img.size(2) == 3, "Expect HxWx3");

        auto img_cpu = img.detach().cpu(); 

        torch::Tensor img_u8;

        if (img_cpu.dtype() == torch::kFloat) {
            img_u8 = img_cpu.clamp(0, 1).mul(255).to(torch::kU8);
        } else {
            img_u8 = img_cpu.to(torch::kU8);
        }

        img_u8 = img_u8.contiguous();

        int H = img_u8.size(0);
        int W = img_u8.size(1);

        stbi_write_png(path.c_str(), W, H, 3, img_u8.data_ptr(), W * 3);
    }

}  // namespace mini_pytorch3d
