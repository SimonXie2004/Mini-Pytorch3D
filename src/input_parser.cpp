#include <mini_pytorch3d/input_parser.h>
#include <mini_pytorch3d/math_utils.h>

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
                sscanf(line, "v %f %f %f", &vertex.x, &vertex.y, &vertex.z);
                vertices.push_back(vertex);
            } else if (line[0] == 'f') {
                float3 face;
                sscanf(line, "f %f %f %f", &face.x, &face.y, &face.z);
                faces.push_back(face);
            }
        }

        fclose(file);

        struct mesh m {
            .vertices = vertices,
            .colors = rand_range(vertices.size()),
            .faces = faces,
            .num_vertices = vertices.size(),
            .num_faces = faces.size()
        };
        return m;
    }
}  // namespace mini_pytorch3d
