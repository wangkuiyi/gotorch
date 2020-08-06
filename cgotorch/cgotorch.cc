// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

char *exception_str(const std::exception &e) {
  auto len = strlen(e.what());
  auto r = new char[len + 1];
  snprintf(r, len, "%s", e.what());
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations
////////////////////////////////////////////////////////////////////////////////

char *RandN(int rows, int cols, int require_grad, Tensor *result) {
  try {
    at::Tensor t = torch::randn(
        {rows, cols}, at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c =
        at::mm(*static_cast<at::Tensor *>(a), *static_cast<at::Tensor *>(b));
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

char *Sum(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(static_cast<at::Tensor *>(a)->sum());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

char *Conv2d(Tensor input, Tensor weight, Tensor bias, int64_t *stride_data,
             int64_t stride_len, int64_t *padding_data, int64_t padding_len,
             int64_t *dilation_data, int64_t dilation_len, int64_t groups,
             Tensor *result) {
  try {
    auto output = at::conv2d(
        *static_cast<at::Tensor *>(input), *static_cast<at::Tensor *>(weight),
        (bias ? *static_cast<at::Tensor *>(bias) : at::Tensor()),
        torch::IntArrayRef(stride_data, stride_len),
        torch::IntArrayRef(padding_data, padding_len),
        torch::IntArrayRef(dilation_data, dilation_len), groups);
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
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

////////////////////////////////////////////////////////////////////////////////
// Backward, Gradient
////////////////////////////////////////////////////////////////////////////////

void Tensor_Backward(Tensor a) { static_cast<at::Tensor *>(a)->backward(); }

Tensor Tensor_Grad(Tensor a) {
  return new at::Tensor(static_cast<at::Tensor *>(a)->grad());
}

////////////////////////////////////////////////////////////////////////////////
// Optimizer
////////////////////////////////////////////////////////////////////////////////

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int nesterov) {
  auto options = torch::optim::SGDOptions(learning_rate)
                     .momentum(momentum)
                     .dampening(dampening)
                     .weight_decay(weight_decay)
                     .nesterov(nesterov);
  return static_cast<torch::optim::Optimizer *>(
      new torch::optim::SGD(std::vector<torch::Tensor>(), options));
}

Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay) {
  auto options = torch::optim::AdamOptions(learning_rate)
                     .betas(std::tuple<double, double>(beta1, beta2))
                     .weight_decay(weight_decay);
  return static_cast<torch::optim::Optimizer *>(
      new torch::optim::Adam(std::vector<torch::Tensor>(), options));
}

void Optimizer_ZeroGrad(Optimizer opt) {
  static_cast<torch::optim::Optimizer *>(opt)->zero_grad();
}

void Optimizer_Step(Optimizer opt) {
  static_cast<torch::optim::Optimizer *>(opt)->step();
}

void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int length) {
  for (int i = 0; i < length; ++i)
    static_cast<torch::optim::Optimizer *>(opt)
        ->param_groups()[0]
        .params()
        .push_back(*(static_cast<torch::Tensor *>(tensors[i])));
}

void Optimizer_Close(Optimizer opt) {
  delete static_cast<torch::optim::Optimizer *>(opt);
}

////////////////////////////////////////////////////////////////////////////////
// Dataset, DataLoader, and Iterator
////////////////////////////////////////////////////////////////////////////////

char *MNIST(const char *data_root, Dataset *dataset) {
  try {
    *dataset = new torch::data::datasets::MNIST(std::string(data_root));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
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

void Iterator_Batch(Iterator iter, Tensor *data, Tensor *target) {
  auto i = *static_cast<TypeIterator *>(iter);
  *data = new at::Tensor(i->data()->data);
  *target = new at::Tensor(i->data()->target);
}

bool Loader_Next(DataLoader loader, Iterator iter) {
  return ++*static_cast<TypeIterator *>(iter) !=
         static_cast<TypeDataLoader *>(loader)->end();
}

void Loader_Close(DataLoader loader) {
  delete static_cast<TypeDataLoader *>(loader);
}
