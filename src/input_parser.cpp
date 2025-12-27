#include <mini_pytorch3d/input_parser.h>
#include <mini_pytorch3d/math_utils.h>
#include <torch/torch.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace mini_pytorch3d {

    struct mesh parse_obj(const std::string& file_path) {
        // Only reads v lines and f lines; Ignores vt, vn, # lines.
        FILE *file = fopen(file_path.c_str(), "r");

        if (file == NULL) {
            perror("Failed to open file");
            return {};
        }

        std::vector<float3> vertices;
        std::vector<float3> faces;  

        char line[128];
        while (fgets(line, sizeof(line), file)) {
            if (line[0] == 'v') {
                float3 vertex;
                sscanf(line, "v %f %f %f", &vertex[0], &vertex[1], &vertex[2]);
                vertices.push_back(vertex);
            } else if (line[0] == 'f') {
                float3 face;
                sscanf(line, "f %f %f %f", &face[0], &face[1], &face[2]);
                faces.push_back(face);
            }
        }

        fclose(file);

        struct mesh m {
            .vertices = torch::from_blob(
                vertices.data(),
                {static_cast<long>(vertices.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone(), 
            .colors = torch::randn(
                {static_cast<long>(vertices.size()), 3}, 
                torch::TensorOptions().dtype(torch::kFloat)
            ), 
            .faces = torch::from_blob(
                faces.data(),
                {static_cast<long>(faces.size()), 3},
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone(),
            .num_vertices = vertices.size(),
            .num_faces = faces.size()
        };
        return m;
    }

    void save_tensor_as_png(const torch::Tensor& img, const std::string& path) {
        TORCH_CHECK(img.device().is_cpu(), "Tensor must be on CPU");
        TORCH_CHECK(img.dim() == 3 && img.size(2) == 3, "Expect HxWx3");

        torch::Tensor img_u8;

        if (img.dtype() == torch::kFloat) {
            img_u8 = img.clamp(0, 1).mul(255).to(torch::kU8);
        } else {
            img_u8 = img.to(torch::kU8);
        }

        img_u8 = img_u8.contiguous();

        int H = img_u8.size(0);
        int W = img_u8.size(1);

        stbi_write_png(path.c_str(), W, H, 3, img_u8.data_ptr(), W * 3);
    }

}  // namespace mini_pytorch3d
