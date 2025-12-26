#include <torch/torch.h>
#include <mini_pytorch3d/test_env.h>
#include <mini_pytorch3d/math_utils.h>
#include <mini_pytorch3d/input_parser.h>
#include <mini_pytorch3d/render.h>
#include <iostream>

int main() {
    // test_torch();

    auto mesh = mini_pytorch3d::parse_obj("data/utah_teapot.obj");
    // auto mesh = mini_pytorch3d::parse_obj("data/stanford_dragon.obj");
    // auto mesh = mini_pytorch3d::parse_obj("data/stanford_bunny.obj");
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

    return 0;
}
