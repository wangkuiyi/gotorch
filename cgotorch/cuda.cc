// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#ifdef WITH_CUDA
#include "c10/cuda/CUDAStream.h"
#endif

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

bool IsCUDAAvailable() { return torch::cuda::is_available(); }

bool IsCUDNNAvailable() { return torch::cuda::cudnn_is_available(); }

const char *CUDA_GetCurrentStream(CUDAStream *stream, Device *device) {
#ifdef WITH_CUDA
  try {
    *stream = static_cast<void *>(new at::cuda::CUDAStream(
        at::cuda::getCurrentCUDAStream((*device)->index())));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
#else
  return exception_str("CUDA API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}
