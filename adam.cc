#include <iostream>

#include "torch/script.h"
#include "torch/optim.h"

int main() {
  int N = 64;
  int D_in = 1000;
  int H = 100;
  int D_out = 10;

  auto x = torch::randn({N, D_in}, at::TensorOptions().requires_grad(false));
  auto y = torch::randn({N, D_out}, at::TensorOptions().requires_grad(false));

  // The Adam optimizer wants parameters in a std::vector.
  std::vector<at::Tensor> params;
  params.push_back(torch::randn({D_in, H},
                                at::TensorOptions().requires_grad(true)));
  params.push_back(torch::randn({H, D_out},
                                at::TensorOptions().requires_grad(true)));

  // Build the optimizer.
  double learning_rate = 1e-3;
  torch::optim::Adam adam(params,
                          torch::optim::AdamOptions(learning_rate));

  // Make quick references for using in the forward pass.
  const at::Tensor & w1 = adam.parameters()[0];
  const at::Tensor & w2 = adam.parameters()[1];

  for (int i = 0; i < 500; ++i) {
    auto y_pred = at::mm(at::clamp(at::mm(x, w1), 0), w2);
    auto loss = at::sum(at::pow(at::sub(y_pred, y), 2));

    if ((i % 100) == 0) {
      std::cout << "loss = " << loss << std::endl;
    }

    adam.zero_grad();
    loss.backward();
    adam.step();
  }
  return 0;
}
