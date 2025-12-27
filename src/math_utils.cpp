#include <mini_pytorch3d/math_utils.h>

namespace mini_pytorch3d {

    torch::Tensor camera_to_mvp_matrix(const camera& cam) {
        torch::Tensor M = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32));

        torch::Tensor V = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor eye = torch::from_blob(
            const_cast<float*>(cam.position.data()),
            {3},
            torch::kFloat
        ).clone();
        torch::Tensor look_at = torch::from_blob(
            const_cast<float*>(cam.look_at.data()),
            {3},
            torch::kFloat
        ).clone();
        torch::Tensor up = torch::from_blob(
            const_cast<float*>(cam.up.data()),
            {3},
            torch::kFloat
        ).clone();

        torch::Tensor f = (look_at - eye);
        f = f / torch::norm(f);
        torch::Tensor s = torch::cross(f, up, /*dim=*/0);
        s = s / torch::norm(s);
        torch::Tensor u = torch::cross(s, f, /*dim=*/0);

        V.index_put_({0, 0}, s[0]);
        V.index_put_({0, 1}, s[1]);
        V.index_put_({0, 2}, s[2]);
        V.index_put_({1, 0}, u[0]);
        V.index_put_({1, 1}, u[1]);
        V.index_put_({1, 2}, u[2]);
        V.index_put_({2, 0}, -f[0]);
        V.index_put_({2, 1}, -f[1]);
        V.index_put_({2, 2}, -f[2]);
        V.index_put_({0, 3}, -torch::dot(s, eye));
        V.index_put_({1, 3}, -torch::dot(u, eye));
        V.index_put_({2, 3}, torch::dot(f, eye));

        float fov_rad = cam.fov * M_PI / 180.0f;
        float tan_half_fov = tan(fov_rad / 2.0f);
        float zn = cam.near_plane;
        float zf = cam.far_plane;

        torch::Tensor P = torch::zeros({4,4}, torch::TensorOptions().dtype(torch::kFloat32));
        P.index_put_({0,0}, 1.0f / (cam.aspect_ratio * tan_half_fov));
        P.index_put_({1,1}, 1.0f / (tan_half_fov));
        P.index_put_({2,2}, -(zf + zn) / (zf - zn));
        P.index_put_({2,3}, -(2.0f * zf * zn) / (zf - zn));
        P.index_put_({3,2}, -1.0f);

        torch::Tensor MVP = torch::matmul(P, torch::matmul(V, M));
        return MVP;
    }

}  // namespace mini_pytorch3d