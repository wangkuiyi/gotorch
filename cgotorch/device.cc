// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

std::unordered_map<std::string, torch::DeviceType> device_type_map = {
    {"cpu", torch::kCPU}, {"cuda", torch::kCUDA}};

const char *Torch_Device(const char *device_type, Device *device) {
  try {
    if (device_type_map.count(std::string(device_type)) == 0) {
      return exception_str("Excepted one of cpu, cuda device type.");
    }
    *device = new torch::Device(device_type_map[std::string(device_type)]);
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
  return nullptr;
}

bool IsCUDAAvailable() { return torch::cuda::is_available(); }

const char *Tensor_To(Tensor input, Device device, int8_t dtype,
                      Tensor *output) {
  try {
    auto result = input->to(*device, static_cast<at::ScalarType>(dtype));
    *output = new at::Tensor(result);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
