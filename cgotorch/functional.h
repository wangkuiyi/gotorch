/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// torch.nn.functional
////////////////////////////////////////////////////////////////////////////////

const char *BatchNorm(Tensor input, Tensor weight, Tensor bias,
                      Tensor running_mean, Tensor running_var, int8_t training,
                      double momentum, double eps, Tensor *result);

const char *Conv2d(Tensor input, Tensor weight, Tensor bias,
                   int64_t *stride_data, int64_t stride_len,
                   int64_t *padding_data, int64_t padding_len,
                   int64_t *dilation_data, int64_t dilation_len, int64_t groups,
                   Tensor *result);

const char *ConvTranspose2d(Tensor input, Tensor weight, Tensor bias,
                            int64_t *stride_data, int64_t stride_len,
                            int64_t *padding_data, int64_t padding_len,
                            int64_t *output_padding_data,
                            int64_t output_padding_len, int64_t groups,
                            int64_t *dilation_data, int64_t dilation_len,
                            Tensor *result);

const char *BinaryCrossEntropy(Tensor input, Tensor target, Tensor weight,
                               const char *reduction, Tensor *result);

const char *CrossEntropy(Tensor input, Tensor target, Tensor weight,
                         int64_t ignore_index, const char *reduction,
                         Tensor *result);

const char *NllLoss(Tensor input, Tensor target, Tensor weight,
                    int64_t ignore_index, const char *reduction,
                    Tensor *result);

const char *FRelu(Tensor input, int8_t inplace, Tensor *result);
const char *FLeakyRelu(Tensor input, double negative_slope, int8_t inplace,
                       Tensor *result);
const char *Linear(Tensor input, Tensor weight, Tensor bias, Tensor *result);

const char *MaxPool2d(Tensor input, int64_t *kernel_data, int64_t kernel_len,
                      int64_t *stride_data, int64_t stride_len,
                      int64_t *padding_data, int64_t padding_len,
                      int64_t *dilation_data, int64_t dilation_len,
                      int8_t ceil_mode, Tensor *result);

const char *AdaptiveAvgPool2d(Tensor input, int64_t *output_size_data,
                              int64_t output_size_len, Tensor *result);

#ifdef __cplusplus
}
#endif
