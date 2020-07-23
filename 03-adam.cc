#include <iostream>

#include "torch/script.h"
#include "torch/optim.h"

int main() {
  int N = 64, D_in = 1000, H = 100, D_out = 10;
  double learning_rate = 1e-3;

  auto x = torch::randn({N, D_in},
                        at::TensorOptions().requires_grad(false));
  auto y = torch::randn({N, D_out},
                        at::TensorOptions().requires_grad(false));

  // The Adam optimizer wants parameters in a std::vector.
  std::vector<at::Tensor> params = {
    torch::randn({D_in, H},
                 at::TensorOptions().requires_grad(true)),
    torch::randn({H, D_out},
                 at::TensorOptions().requires_grad(true))
  };

  // Build the optimizer.
  torch::optim::Adam adam(params,
                          torch::optim::AdamOptions(learning_rate));

  // Make quick references for using in the forward pass.
  const at::Tensor & w1 = adam.parameters()[0];
  const at::Tensor & w2 = adam.parameters()[1];

  for (int i = 0; i < 500; ++i) {
    auto y_pred = at::mm(at::clamp(at::mm(x, w1), 0), w2);
    auto loss = at::sum(at::pow(at::sub(y_pred, y), 2));

    if ((i % 100) == 99) {
      std::cout << "loss = " << loss << std::endl;
    }

    adam.zero_grad();
    loss.backward();
    adam.step();
  }
  return 0;
}
