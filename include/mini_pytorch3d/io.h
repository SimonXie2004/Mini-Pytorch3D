#ifndef MINI_PYTORCH3D_IO_H
#define MINI_PYTORCH3D_IO_H

#include <torch/types.h>
#include <string>

namespace mini_pytorch3d {
    
    struct mesh parse_obj(const std::string& file_path);

    void save_tensor_as_png(const torch::Tensor& img, const std::string& path); 

}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_IO_H