/* Copyright 2020, GoTorch Authors */

#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif
const char *Gloo_NewFileStore(const char *path, int64_t num_workers,
                              Store *store);

const char *Gloo_NewTCPStore(const char *addr, int64_t port,
                             int64_t num_workers, int64_t is_server,
                             Store *store);

const char *Gloo_DeleteStore(Store store);

const char *Gloo_NewProcessGroupGloo(Store store, int64_t rank, int64_t size,
                                     ProcessGroupGloo *pg);

const char *Gloo_DeleteProcessGroupGloo(ProcessGroupGloo pg);

const char *Gloo_allreduce(ProcessGroupGloo pg, Tensor *tensors,
                           int64_t length);

const char *Gloo_allreduce_coalesced(ProcessGroupGloo pg, Tensor *tensors,
                                     int64_t length);

const char *Gloo_broadcast(ProcessGroupGloo pg, Tensor *tensors,
                           int64_t length);

#ifdef __cplusplus
}
#endif
