/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Optimizer torch.optim
////////////////////////////////////////////////////////////////////////////////

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int64_t nesterov);
Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay);

void Optimizer_ZeroGrad(Optimizer opt);
void Optimizer_Step(Optimizer opt);
void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int64_t length);
void Optimizer_SetLR(Optimizer opt, double learning_rate);
void Optimizer_Close(Optimizer opt);

#ifdef __cplusplus
}
#endif
