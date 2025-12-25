#include <torch/torch.h>

namespace mini_pytorch3d {

    int test_torch() {
        torch::Tensor x = torch::tensor(
            {0.0},
            torch::TensorOptions().requires_grad(true)
        );

        torch::optim::SGD optimizer(
            {x},
            torch::optim::SGDOptions(0.1)
        );

        for (int i = 0; i < 20; i++) {
            optimizer.zero_grad();

            torch::Tensor loss = torch::pow(x - 3.0, 2);

            loss.backward();
            optimizer.step();

            std::cout
                << "Iter " << i
                << " | x = " << x.item<double>()
                << " | loss = " << loss.item<double>()
                << std::endl;
        }

        return 0;
    }

} // namespace mini_pytorch3d
