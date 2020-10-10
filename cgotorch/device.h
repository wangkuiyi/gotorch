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

#ifdef __cplusplus
}
#endif
