// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const std::exception &e) {
  auto len = strlen(e.what());
  auto r = new char[len + 1];
  snprintf(r, len, "%s", e.what());
  return r;
}

std::unordered_map<std::string, torch::nn::init::FanModeType> fan_mode_map = {
    {"fan_in", torch::kFanIn},
    {"fan_out", torch::kFanOut},
};

std::unordered_map<std::string, torch::nn::init::NonlinearityType>
    non_linearity_map = {
        {"relu", torch::kReLU},
        {"leaky_relu", torch::kLeakyReLU},
        {"tanh", torch::kTanh},
        {"sigmoid", torch::kSigmoid},
        {"linear", torch::kLinear},
        {"conv1d", torch::kConv1D},
        {"conv2d", torch::kConv2D},
        {"conv3d", torch::kConv3D},
        {"conv_transpose1d", torch::kConvTranspose1D},
        {"conv_transpose2d", torch::kConvTranspose2D},
        {"conv_transpose3d", torch::kConvTranspose3D},
};

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations
////////////////////////////////////////////////////////////////////////////////

const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::randn(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::empty(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Zeros_(Tensor *tensor) {
  try {
    torch::nn::init::zeros_(**tensor);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Uniform_(Tensor *tensor, double low, double high) {
  try {
    torch::nn::init::uniform_(**tensor, low, high);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *KaimingUniform_(double a, const char *fan_mod,
                            const char *non_linearity, Tensor *tensor) {
  try {
    torch::nn::init::kaiming_uniform_(
        **tensor, a, fan_mode_map[std::string(fan_mod)],
        non_linearity_map[std::string(non_linearity)]);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *CalculateFanInAndFanOut(Tensor tensor, int64_t *fan_in,
                                    int64_t *fan_out) {
  try {
    const auto &res = torch::nn::init::_calculate_fan_in_and_fan_out(*tensor);
    *fan_in = std::get<0>(res);
    *fan_out = std::get<1>(res);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c = at::mm(*a, *b);
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Sum(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Conv2d(Tensor input, Tensor weight, Tensor bias,
                   int64_t *stride_data, int64_t stride_len,
                   int64_t *padding_data, int64_t padding_len,
                   int64_t *dilation_data, int64_t dilation_len, int64_t groups,
                   Tensor *result) {
  try {
    auto output =
        at::conv2d(*input, *weight, (bias ? *bias : at::Tensor()),
                   torch::IntArrayRef(stride_data, stride_len),
                   torch::IntArrayRef(padding_data, padding_len),
                   torch::IntArrayRef(dilation_data, dilation_len), groups);
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *BatchNorm(Tensor input, Tensor weight, const Tensor *bias,
                      const Tensor *running_mean, const Tensor *running_var,
                      int8_t training, double momentum, double eps,
                      int8_t cudnn_enabled, Tensor *result) {
  try {
    auto output =
        at::batch_norm(*input, *weight, (bias ? **bias : at::Tensor()),
                       (running_mean ? **running_mean : at::Tensor()),
                       (running_var ? **running_var : at::Tensor()), training,
                       momentum, eps, cudnn_enabled);
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Relu(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->relu());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result) {
  try {
    *result = new at::Tensor(at::leaky_relu(*a, negative_slope));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Tanh(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->tanh());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *Sigmoid(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sigmoid());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *LogSoftmax(Tensor a, int dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->log_softmax(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

const char *ConvTranspose2d(Tensor input, Tensor weight, Tensor bias,
                            int64_t *stride_data, int64_t stride_len,
                            int64_t *padding_data, int64_t padding_len,
                            int64_t *output_padding_data,
                            int64_t output_padding_len, int64_t groups,
                            int64_t *dilation_data, int64_t dilation_len,
                            Tensor *result) {
  try {
    auto output = at::conv_transpose2d(
        *input, *weight, (bias ? *bias : at::Tensor()),
        torch::IntArrayRef(stride_data, stride_len),
        torch::IntArrayRef(padding_data, padding_len),
        torch::IntArrayRef(output_padding_data, output_padding_len), groups,
        torch::IntArrayRef(dilation_data, dilation_len));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e);
  }
}

void Tensor_Print(Tensor a) { std::cout << *a << std::endl; }

void Tensor_Close(Tensor a) { delete a; }

// The caller must free the returned string by calling FreeString.
const char *Tensor_String(Tensor a) {
  std::stringstream ss;
  ss << *a;
  std::string s = ss.str();
  char *r = new char[s.size() + 1];
  snprintf(r, s.size(), "%s", s.c_str());
  return r;
}

void FreeString(const char *s) { delete[] s; }

////////////////////////////////////////////////////////////////////////////////
// Backward, Gradient
////////////////////////////////////////////////////////////////////////////////

void Tensor_Backward(Tensor a) { a->backward(); }

Tensor Tensor_Grad(Tensor a) { return new at::Tensor(a->grad()); }

////////////////////////////////////////////////////////////////////////////////
// Optimizer
////////////////////////////////////////////////////////////////////////////////

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int64_t nesterov) {
  auto options = torch::optim::SGDOptions(learning_rate)
                     .momentum(momentum)
                     .dampening(dampening)
                     .weight_decay(weight_decay)
                     .nesterov(nesterov);
  return new torch::optim::SGD(std::vector<torch::Tensor>(), options);
}

Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay) {
  auto options = torch::optim::AdamOptions(learning_rate)
                     .betas(std::tuple<double, double>(beta1, beta2))
                     .weight_decay(weight_decay);
  return new torch::optim::Adam(std::vector<torch::Tensor>(), options);
}

void Optimizer_ZeroGrad(Optimizer opt) { opt->zero_grad(); }

void Optimizer_Step(Optimizer opt) { opt->step(); }

void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int64_t length) {
  for (int64_t i = 0; i < length; ++i)
    opt->param_groups()[0].params().push_back(*tensors[i]);
}

void Optimizer_Close(Optimizer opt) { delete opt; }

////////////////////////////////////////////////////////////////////////////////
// Dataset, DataLoader, and Iterator
////////////////////////////////////////////////////////////////////////////////

const char *MNIST(const char *data_root, Dataset *dataset) {
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

DataLoader MakeDataLoader(Dataset dataset, int64_t batchsize) {
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
