#include <mini_pytorch3d/math_utils.h>

#include <cstdlib>
#include <ctime>

namespace mini_pytorch3d {
    std::vector<float3> rand_range(std::size_t n) {
        std::vector<float3> result;
        result.reserve(n);
        std::srand(static_cast<unsigned>(std::time(0)));
        for (std::size_t i = 0; i < n; ++i) {
            float3 vec = {static_cast<float>(std::rand()) / RAND_MAX,
                          static_cast<float>(std::rand()) / RAND_MAX,
                          static_cast<float>(std::rand()) / RAND_MAX};
            result.push_back(vec);
        }
        return result;
    }
}  // namespace mini_pytorch3d