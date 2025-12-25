#include <torch/torch.h>
#include <mini_pytorch3d/test_env.h>
#include <mini_pytorch3d/math_utils.h>
#include <mini_pytorch3d/input_parser.h>
#include <iostream>

int main() {
    // test_torch();

    auto mesh = mini_pytorch3d::parse_obj("data/utah_teapot.obj");
    std::cout << "Parsed mesh with " << mesh.num_vertices << " vertices and "
              << mesh.num_faces << " faces." << std::endl;

    return 0;
}
