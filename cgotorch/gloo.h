/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif
const char *Gloo_NewFileStore(const char *path, int64_t num_workers,
                              FileStore *store);
const char *Gloo_NewProcessGroupGloo(FileStore store, int64_t rank,
                                     int64_t size, ProcessGroupGloo *pg);

const char *Gloo_allreduce(ProcessGroupGloo pg, Tensor *tensors,
                           int64_t length);
#ifdef __cplusplus
}
#endif