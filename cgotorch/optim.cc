// Copyright 2020, GoTorch Authors
#include "cgotorch/optim.h"

#include <vector>

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int64_t nesterov) {
  return new torch::optim::SGD(std::vector<torch::optim::OptimizerParamGroup>{},
                               torch::optim::SGDOptions(learning_rate)
                                   .momentum(momentum)
                                   .dampening(dampening)
                                   .weight_decay(weight_decay)
                                   .nesterov(nesterov));
}

Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay) {
  auto options = torch::optim::AdamOptions(learning_rate)
                     .betas(std::make_tuple(beta1, beta2))
                     .weight_decay(weight_decay);
  return new torch::optim::Adam(std::vector<torch::Tensor>(), options);
}

void Optimizer_ZeroGrad(Optimizer opt) { opt->zero_grad(); }

void Optimizer_Step(Optimizer opt) { opt->step(); }

void Optimizer_AddParameters(Optimizer opt, Tensor* tensors, int64_t length) {
  std::vector<torch::Tensor> params;
  while (params.size() < length) params.push_back(**tensors++);
  opt->add_param_group({params});
}

void Optimizer_SetLR(Optimizer opt, double learning_rate) {
  if (dynamic_cast<torch::optim::SGD*>(opt)) {
    for (auto& pg : opt->param_groups()) {
      static_cast<torch::optim::SGDOptions*>(&pg.options())->lr(learning_rate);
    }
  } else if (dynamic_cast<torch::optim::Adam*>(opt)) {
    for (auto& pg : opt->param_groups()) {
      static_cast<torch::optim::AdamOptions*>(&pg.options())->lr(learning_rate);
    }
  }
}

void Optimizer_Close(Optimizer opt) { delete opt; }
