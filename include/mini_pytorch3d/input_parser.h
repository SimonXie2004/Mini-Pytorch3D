#ifndef MINI_PYTORCH3D_INPUT_PARSER_H
#define MINI_PYTORCH3D_INPUT_PARSER_H

#include <string>
#include <vector>

namespace mini_pytorch3d {
    struct mesh parse_obj(const std::string& file_path);
}  // namespace mini_pytorch3d

#endif // MINI_PYTORCH3D_INPUT_PARSER_H