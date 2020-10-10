/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif
////////////////////////////////////////////////////////////////////////////////
// Nvidia CUDA
////////////////////////////////////////////////////////////////////////////////
bool IsCUDAAvailable();
bool IsCUDNNAvailable();
const char *CUDA_GetCurrentCUDAStream(CUDAStream *stream, Device *device);
const char *CUDA_SetCurrentCUDAStream(CUDAStream stream);
const char *CUDA_GetCUDAStreamFromPool(CUDAStream *stream, Device *device);
const char *CUDA_Synchronize(CUDAStream stream);
const char *CUDA_Query(CUDAStream stream, int8_t *result);
#ifdef __cplusplus
}
#endif
