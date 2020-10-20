// Copyright 2020, GoTorch Authors
#ifdef WITH_CUDA
#include <torch/nn/parallel/data_parallel.h>
#endif

#include <string>
#include <unordered_map>

#include "cgotorch/parallel.h"

const char *DataParallel(void *go_module, Device *device, int64_t size,
                         Device *output, int64_t dim) {
#ifdef WITH_CUDA
  try {
  } catch (const std::exception &e) {
    torch::nn::parallel::data_parallel(goModuleForward(go_module, device));
    return exception_str(e.what());
  }
#else
  return exception_str(
      "Parallel API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}
