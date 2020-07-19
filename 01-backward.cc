#include <iostream>

// Typical you only need include the all-in-one header: #include <torch/script.h>

// Here we only use ATen's functional header:
#include <ATen/Functions.h>

// We need use variable factory methods:
#include <torch/csrc/autograd/generated/variable_factories.h>

int main() {
  // We could use at::randn(...) to create a Tensor.
  // We could also use torch::randn(...) to create a Variable.
  // The difference is that Variable support autograd but Tensor doesn't.
  //
  // Variable and Tensor used to be two difference classes. We already made
  // an effort to consolidate them - now Variable class is Tensor class with
  // additional autograd metadata. But the factory methods haven't been
  // consolidated yet.
  auto a = torch::randn({3, 4}, at::TensorOptions().requires_grad(true));
  std::cout << "a = " << a << std::endl;

  auto b = torch::randn({4, 1}, at::TensorOptions().requires_grad(true));
  std::cout << "b = " << b << std::endl;

  auto c = at::mm(a, b);
  std::cout << "c = " << c << std::endl;

  auto d = c.sum();
  std::cout << "d = " << d << std::endl;

  d.backward();

  std::cout << "a.grad = " << a.grad() << std::endl;
  std::cout << "b.grad = " << b.grad() << std::endl;
  return 0;
}
