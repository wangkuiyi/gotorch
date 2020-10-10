// Copyright 2020, GoTorch Authors
#ifdef WITH_CUDA
#include "c10/cuda/CUDAStream.h"
#endif

#include "cgotorch/cuda.h"

bool IsCUDAAvailable() { return torch::cuda::is_available(); }

bool IsCUDNNAvailable() { return torch::cuda::cudnn_is_available(); }

const char *CUDA_GetCurrentCUDAStream(CUDAStream *stream, Device *device) {
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

const char *CUDA_GetCUDAStreamFromPool(CUDAStream *stream, Device *device) {
#ifdef WITH_CUDA
  try {
    *stream = static_cast<void *>(
        new at::cuda::CUDAStream(at::cuda::getStreamFromPool(
            false /**isHighPriority**/, (*device)->index())));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
#else
  return exception_str("CUDA API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}

const char *CUDA_SetCurrentCUDAStream(CUDAStream stream) {
#ifdef WITH_CUDA
  try {
    at::cuda::setCurrentCUDAStream(
        *static_cast<at::cuda::CUDAStream *>(stream));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
#else
  return exception_str("CUDA API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}

const char *CUDA_Synchronize(CUDAStream stream) {
#ifdef WITH_CUDA
  try {
    static_cast<at::cuda::CUDAStream *>(stream)->synchronize();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
#else
  return exception_str("CUDA API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}

const char *CUDA_Query(CUDAStream stream, int8_t *result) {
#ifdef WITH_CUDA
  try {
    *result = static_cast<at::cuda::CUDAStream *>(stream)->query() ? 1 : 0;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
#else
  return exception_str("CUDA API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}
