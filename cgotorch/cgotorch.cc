#include "cgotorch.h"

#include <iostream>

#include "torch/script.h"

Tensor RandN(int rows, int cols, int require_grad) {
  at::Tensor t = torch::randn({rows,cols},
			      at::TensorOptions().requires_grad(require_grad));
  return new at::Tensor(std::move(t));
}

Tensor MM(Tensor a, Tensor b) {
  at::Tensor c = at::mm(*static_cast<at::Tensor*>(a),
			*static_cast<at::Tensor*>(b));
  return new at::Tensor(std::move(c));
}

Tensor Sum(Tensor a) {
  at::Tensor r = static_cast<at::Tensor*>(a)->sum();
  return new at::Tensor(std::move(r));
}

void Backward(Tensor a) {
  static_cast<at::Tensor*>(a)->backward();
}

Tensor Grad(Tensor a) {
  at::Tensor r = static_cast<at::Tensor*>(a)->grad();
  return new at::Tensor(std::move(r));
}

void PrintTensor(Tensor a) {
  std::cout << *static_cast<at::Tensor*>(a) << std::endl;
}

const char* Tensor_String(Tensor a) {
  return static_cast<at::Tensor*>(a)->toString().c_str()
}
