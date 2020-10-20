/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Parallel
////////////////////////////////////////////////////////////////////////////////

const char *DataParallel(void *go_module, Device *device, int64_t size,
                         Device *output, int64_t dim);

#ifdef __cplusplus
}
#endif
