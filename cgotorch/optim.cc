// Copyright 2020, GoTorch Authors
#include <tuple>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

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
