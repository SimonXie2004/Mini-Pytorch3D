#include <torch/torch.h>
#include <mini_pytorch3d/test_env.h>
#include <mini_pytorch3d/math_utils.h>
#include <mini_pytorch3d/input_parser.h>
#include <mini_pytorch3d/render.h>
#include <iostream>

// ./build/src/mini_pytorch3d data/stanford_dragon.obj output/output.png
int main(int argc, char **argv) {
    // test_torch();

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_obj> <output_image_path>" << std::endl;
        return -1;
    }

    auto mesh = mini_pytorch3d::parse_obj(argv[1]);
    std::cout << "Parsed mesh with " << mesh.vertices.size(0) << " vertices and "
              << mesh.faces.size(0) << " faces." << std::endl;

    auto diffrast_image = mini_pytorch3d::render(mesh, 
        mini_pytorch3d::camera{
            .position = {0.0f, 0.0f, 3.0f},
            .look_at = {0.0f, 0.0f, 0.0f},
            .up = {0.0f, 1.0f, 0.0f},
            .fov = 60.0f,
            .aspect_ratio = 1.0f,
            .near_plane = 0.1f,
            .far_plane = 10.0f,
            .H = 512,
            .W = 512
        },
        mini_pytorch3d::diffrast_args{
            .sigma = 1e-5f,
            .gamma = 1e-4f
        }
    );
    std::cout << "Rendered image size: " << diffrast_image.sizes() << std::endl;

    mini_pytorch3d::save_tensor_as_png(diffrast_image, argv[2]);

    return 0;
}
