/* Copyright 2020, GoTorch Authors */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#include <torch/torch.h>
#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupGloo.hpp>

#include <vector>  // NOLINT
extern "C" {
typedef at::Tensor *Tensor;
typedef torch::optim::Optimizer *Optimizer;
typedef torch::data::datasets::MNIST *MNIST;
typedef torch::data::transforms::Normalize<> *Normalize;
typedef torch::Device *Device;
typedef std::vector<char> *ByteBuffer;  // NOLINT
typedef c10d::FileStore *FileStore;
typedef c10d::ProcessGroupGloo *ProcessGroupGloo;
#else
typedef void *Tensor;
typedef void *Optimizer;
typedef void *MNIST;
typedef void *Normalize;
typedef void *Device;
typedef void *ByteBuffer;
typedef void *FileStore;
typedef void *ProcessGroupGloo;
#endif
typedef void *CUDAStream;

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const char *e);
#ifdef __cplusplus
}
#endif
