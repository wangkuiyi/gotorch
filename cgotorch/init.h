/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// torch.nn.init
////////////////////////////////////////////////////////////////////////////////

// torch.nn.init.zeros_
const char *Zeros_(Tensor *tensor);
// torch.nn.init.ones_
const char *Ones_(Tensor *tensor);
// torch.nn.init.uniform_
const char *Uniform_(Tensor *tensor, double low, double high);
// torch.nn.init.normal_
const char *Normal_(Tensor *tensor, double mean, double std);
// torch.nn.init.kaiming_uniform_
const char *KaimingUniform_(double a, const char *fan_mod,
                            const char *non_linearity, Tensor *tensor);
const char *CalculateFanInAndFanOut(Tensor tensor, int64_t *fan_in,
                                    int64_t *fan_out);

void ManualSeed(int64_t seed);

#ifdef __cplusplus
}
#endif
