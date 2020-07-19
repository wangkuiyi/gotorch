#include <iostream>

#include "torch/script.h"

int main() {
  int N = 64;
  int D_in = 1000;
  int H = 100;
  int D_out = 10;

  auto x = torch::randn({N, D_in}, at::TensorOptions().requires_grad(false));
  auto y = torch::randn({N, D_out}, at::TensorOptions().requires_grad(false));

  auto w1 = torch::randn({D_in, H}, at::TensorOptions().requires_grad(true));
  auto w2 = torch::randn({H, D_out}, at::TensorOptions().requires_grad(true));

  double learning_rate = 1e-6;

  for (int i = 0; i < 500; ++i) {
    auto y_pred = at::mm(at::clamp(at::mm(x, w1), 0), w2);
    auto loss = at::sum(at::pow(at::sub(y_pred, y), 2));

    if ((i % 100) == 0) {
      std::cout << "loss = " << loss << std::endl;
    }

    loss.backward();

    at::NoGradGuard guard;
    w1.sub_(w1.grad(), learning_rate);
    w2.sub_(w2.grad(), learning_rate);
    w1.grad().zero_();
    w2.grad().zero_();
  }
  return 0;
}
