// Copyright 2020, GoTorch Authors
#include "cgotorch/init.h"

#include <string>
#include <unordered_map>

std::unordered_map<std::string, torch::nn::init::FanModeType> fan_mode_map = {
    {"fan_in", torch::kFanIn},
    {"fan_out", torch::kFanOut},
};

std::unordered_map<std::string, torch::nn::init::NonlinearityType>
    non_linearity_map = {
        {"relu", torch::kReLU},
        {"leaky_relu", torch::kLeakyReLU},
        {"tanh", torch::kTanh},
        {"sigmoid", torch::kSigmoid},
        {"linear", torch::kLinear},
        {"conv1d", torch::kConv1D},
        {"conv2d", torch::kConv2D},
        {"conv3d", torch::kConv3D},
        {"conv_transpose1d", torch::kConvTranspose1D},
        {"conv_transpose2d", torch::kConvTranspose2D},
        {"conv_transpose3d", torch::kConvTranspose3D},
};

const char *Zeros_(Tensor *tensor) {
  try {
    torch::nn::init::zeros_(**tensor);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Ones_(Tensor *tensor) {
  try {
    torch::nn::init::ones_(**tensor);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Uniform_(Tensor *tensor, double low, double high) {
  try {
    torch::nn::init::uniform_(**tensor, low, high);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Normal_(Tensor *tensor, double mean, double std) {
  try {
    torch::nn::init::normal_(**tensor, mean, std);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *KaimingUniform_(double a, const char *fan_mod,
                            const char *non_linearity, Tensor *tensor) {
  try {
    torch::nn::init::kaiming_uniform_(
        **tensor, a, fan_mode_map[std::string(fan_mod)],
        non_linearity_map[std::string(non_linearity)]);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *CalculateFanInAndFanOut(Tensor tensor, int64_t *fan_in,
                                    int64_t *fan_out) {
  try {
    const auto &res = torch::nn::init::_calculate_fan_in_and_fan_out(*tensor);
    *fan_in = std::get<0>(res);
    *fan_out = std::get<1>(res);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

void ManualSeed(int64_t seed) { torch::manual_seed(seed); }
