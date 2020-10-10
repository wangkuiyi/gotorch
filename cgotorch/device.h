/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Device
////////////////////////////////////////////////////////////////////////////////

const char *Torch_Device(const char *device_type, Device *device);
void SetNumThreads(int32_t n);

const char *Tensor_To(Tensor input, Device device, int8_t dtype,
                      Tensor *output);
const char *Tensor_CastTo(Tensor input, int8_t dtype, Tensor *output);
const char *Tensor_CopyTo(Tensor input, Device device, Tensor *output);
const char *Tensor_PinMemory(Tensor input, Tensor *output);
const char *Tensor_CUDA(Tensor input, Device device, int8_t non_blocking,
                        Tensor *output);

#ifdef __cplusplus
}
#endif
