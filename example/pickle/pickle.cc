// Copyright 2020. GoTorch Authors.
#include <iostream>

#include "torch/csrc/api/include/torch/serialize.h"
#include "torch/script.h"

int main() {
  auto a = torch::randn({3, 4}, at::TensorOptions().requires_grad(false));
  std::cout << "Generated tensor = " << a << std::endl;

  auto e = torch::pickle_save(a);
  std::cout << "Encoded buffer size = " << e.size() << std::endl;

  auto v = torch::pickle_load(e);
  std::cout << "Loaded tensor = " << v << std::endl;
  return 0;
}
