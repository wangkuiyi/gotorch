// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

Tensor RandN(int rows, int cols, int require_grad) {
  at::Tensor t = torch::randn({rows, cols},
                              at::TensorOptions().requires_grad(require_grad));
  return new at::Tensor(t);
}

Tensor MM(Tensor a, Tensor b) {
  at::Tensor c =
      at::mm(*static_cast<at::Tensor *>(a), *static_cast<at::Tensor *>(b));
  return new at::Tensor(c);
}

Tensor Sum(Tensor a) {
  return new at::Tensor(static_cast<at::Tensor *>(a)->sum());
}

void Tensor_Backward(Tensor a) { static_cast<at::Tensor *>(a)->backward(); }

Tensor Tensor_Grad(Tensor a) {
  return new at::Tensor(static_cast<at::Tensor *>(a)->grad());
}

void Tensor_Print(Tensor a) {
  std::cout << *static_cast<at::Tensor *>(a) << std::endl;
}

void Tensor_Close(Tensor a) { delete static_cast<at::Tensor *>(a); }

// The caller must free the returned string by calling FreeString.
const char *Tensor_String(Tensor a) {
  std::stringstream ss;
  ss << *static_cast<at::Tensor *>(a);
  std::string s = ss.str();
  char *r = new char[s.size() + 1];
  snprintf(r, s.size(), "%s", s.c_str());
  return r;
}

void FreeString(const char *s) { delete[] s; }

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int nesterov) {
  auto options = torch::optim::SGDOptions(learning_rate)
                     .momentum(momentum)
                     .dampening(dampening)
                     .weight_decay(weight_decay)
                     .nesterov(nesterov);
  return static_cast<torch::optim::Optimizer*>(
      new torch::optim::SGD(std::vector<torch::Tensor>(), options));
}

void ZeroGrad(Optimizer opt) {
  static_cast<torch::optim::Optimizer *>(opt)->zero_grad();
}

void Step(Optimizer opt) { static_cast<torch::optim::Optimizer *>(opt)->step(); }

void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int length) {
  for (int i = 0; i < length; ++i)
    static_cast<torch::optim::Optimizer *>(opt)->param_groups()[0].params().push_back(
        *(static_cast<torch::Tensor *>(tensors[i])));
}

void Optimizer_Close(Optimizer opt) {
  delete static_cast<torch::optim::Optimizer *>(opt);
}

Dataset MNIST(const char *data_root) {
  return new torch::data::datasets::MNIST(std::string(data_root));
}

void MNIST_Close(Dataset d) {
  delete static_cast<torch::data::datasets::MNIST *>(d);
}

Transform Normalize(double mean, double stddev) {
  return new torch::data::transforms::Normalize<>(mean, stddev);
}

Transform Stack() { return new torch::data::transforms::Stack<>(); }

void Dataset_Normalize(Dataset dataset, Transform transform) {
  static_cast<torch::data::datasets::MNIST *>(dataset)->map(
      *(static_cast<torch::data::transforms::Normalize<> *>(transform)));
}

void Dataset_Stack(Dataset dataset, Transform transform) {
  static_cast<torch::data::datasets::MNIST *>(dataset)->map(
      *(static_cast<torch::data::transforms::Stack<> *>(transform)));
}

using TypeDataLoader =
    torch::data::StatelessDataLoader<torch::data::datasets::MNIST,
                                     torch::data::samplers::SequentialSampler>;

using TypeIterator = torch::data::Iterator<TypeDataLoader::BatchType>;

DataLoader MakeDataLoader(Dataset dataset, int batchsize) {
  auto p = torch::data::make_data_loader(
      *(static_cast<torch::data::datasets::MNIST *>(dataset)), batchsize);
  return p.release();
}

Iterator Loader_Begin(DataLoader loader) {
  return new TypeIterator(static_cast<TypeDataLoader *>(loader)->begin());
}

void Loader_Data(Iterator iter, Tensor array[]) {
  array[0] = new at::Tensor((*static_cast<TypeIterator *>(iter))->data()->data);
  array[1] = new at::Tensor((*static_cast<TypeIterator *>(iter))->data()->target);
}

bool Loader_Next(DataLoader loader, Iterator iter) {
  return ++*static_cast<TypeIterator *>(iter) !=
         static_cast<TypeDataLoader *>(loader)->end();
}

void Loader_Close(DataLoader loader) {
  delete static_cast<TypeDataLoader *>(loader);
}
