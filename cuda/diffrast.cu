#include <torch/torch.h>
#include <mini_pytorch3d/math_utils.h>

namespace mini_pytorch3d {

    // Implementation of NON-differentiable rasterization (unrelated, just for fun)
    torch::Tensor rast(const triangle_buffer& tb, int image_width, int image_height) {

        return torch::zeros({image_height, image_width, 3}, torch::kU8); // Placeholder
    }

    // Implementation of differentiable rasterization
    torch::Tensor diffrast(const triangle_buffer& tb, const diffrast_args& args, 
                            int image_width, int image_height) {
        

        return torch::zeros({image_height, image_width, 3}, torch::kU8); // Placeholder
    }

} // namespace mini_pytorch3d
