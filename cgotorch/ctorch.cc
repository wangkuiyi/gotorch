#include "ctorch.h"

#include "torch/script.h"

Tensor RandN(int rows, int cols, int require_grad) {
  at::Tensor t = torch::randn({rows,cols},
			      at::TensorOptions().requires_grad(require_grad));
  at::Tensor* wrapper = new at::Tensor(std::move(t));
  return static_cast<Tensor>(wrapper);
}

Tensor MM(Tensor a, Tensor b) {
  at::Tensor c = at::mm(*static_cast<at::Tensor*>(a),
			*static_cast<at::Tensor*>(b));
  at::Tensor* wrapper = new at::Tensor(std::move(c));
  return static_cast<Tensor>(wrapper);
}

Tensor Sum(Tensor a) {
  at::Tensor r = static_cast<at::Tensor*>(a)->sum();
  at::Tensor* wrapper = new at::Tensor(std::move(r));
  return static_cast<Tensor>(wrapper);
}
