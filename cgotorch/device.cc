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

void SetNumThreads(int32_t n) { torch::set_num_threads(n); }
