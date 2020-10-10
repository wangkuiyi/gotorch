// Copyright 2020, GoTorch Authors
#include "cgotorch/functional.h"

#include <string>
#include <unordered_map>
#include <vector>

const char *BatchNorm(Tensor input, Tensor weight, Tensor bias,
                      Tensor running_mean, Tensor running_var, int8_t training,
                      double momentum, double eps, Tensor *result) {
  try {
    auto output = torch::nn::functional::batch_norm(
        *input, (running_mean ? *running_mean : at::Tensor()),
        (running_var ? *running_var : at::Tensor()),
        torch::nn::functional::BatchNormFuncOptions()
            .weight(weight ? *weight : at::Tensor())
            .bias(bias ? *bias : at::Tensor())
            .training(training)
            .momentum(momentum)
            .eps(eps));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Conv2d(Tensor input, Tensor weight, Tensor bias,
                   int64_t *stride_data, int64_t stride_len,
                   int64_t *padding_data, int64_t padding_len,
                   int64_t *dilation_data, int64_t dilation_len, int64_t groups,
                   Tensor *result) {
  try {
    auto output = torch::nn::functional::conv2d(
        *input, *weight,
        torch::nn::functional::Conv2dFuncOptions()
            .bias(bias ? *bias : at::Tensor())
            .stride(torch::IntArrayRef(stride_data, stride_len))
            .padding(torch::IntArrayRef(padding_data, padding_len))
            .dilation(torch::IntArrayRef(dilation_data, dilation_len))
            .groups(groups));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *ConvTranspose2d(Tensor input, Tensor weight, Tensor bias,
                            int64_t *stride_data, int64_t stride_len,
                            int64_t *padding_data, int64_t padding_len,
                            int64_t *output_padding_data,
                            int64_t output_padding_len, int64_t groups,
                            int64_t *dilation_data, int64_t dilation_len,
                            Tensor *result) {
  try {
    auto output = torch::nn::functional::conv_transpose2d(
        *input, *weight,
        torch::nn::functional::ConvTranspose2dFuncOptions()
            .bias(bias ? *bias : at::Tensor())
            .stride(torch::IntArrayRef(stride_data, stride_len))
            .padding(torch::IntArrayRef(padding_data, padding_len))
            .output_padding(
                torch::IntArrayRef(output_padding_data, output_padding_len))
            .groups(groups)
            .dilation(torch::IntArrayRef(dilation_data, dilation_len)));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *BinaryCrossEntropy(Tensor input, Tensor target, Tensor weight,
                               const char *reduction, Tensor *result) {
  static std::unordered_map<std::string, torch::nn::BCELossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  try {
    auto output = torch::nn::functional::binary_cross_entropy(
        *input, *target,
        torch::nn::functional::BinaryCrossEntropyFuncOptions()
            .weight((weight ? *weight : torch::Tensor()))
            .reduction(reduce_map[std::string(reduction)]));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *CrossEntropy(Tensor input, Tensor target, Tensor weight,
                         int64_t ignore_index, const char *reduction,
                         Tensor *result) {
  static std::unordered_map<std::string,
                            torch::nn::CrossEntropyLossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  try {
    auto output = torch::nn::functional::cross_entropy(
        *input, *target,
        torch::nn::functional::CrossEntropyFuncOptions()
            .weight((weight ? *weight : torch::Tensor()))
            .ignore_index(ignore_index)
            .reduction(reduce_map[std::string(reduction)]));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *FRelu(Tensor input, int8_t inplace, Tensor *result) {
  try {
    auto out = torch::nn::functional::relu(
        *input, torch::nn::functional::ReLUFuncOptions(inplace));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *FLeakyRelu(Tensor input, double negative_slope, int8_t inplace,
                       Tensor *result) {
  try {
    auto out = torch::nn::functional::leaky_relu(
        *input, torch::nn::functional::LeakyReLUFuncOptions()
                    .negative_slope(negative_slope)
                    .inplace(inplace));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *NllLoss(Tensor input, Tensor target, Tensor weight,
                    int64_t ignore_index, const char *reduction,
                    Tensor *result) {
  static std::unordered_map<std::string, torch::nn::NLLLossOptions::reduction_t>
      reduce_map = {
          {"none", torch::kNone},
          {"mean", torch::kMean},
          {"sum", torch::kSum},
      };
  try {
    auto output = torch::nn::functional::nll_loss(
        *input, *target,
        torch::nn::functional::NLLLossFuncOptions()
            .weight((weight ? *weight : torch::Tensor()))
            .ignore_index(ignore_index)
            .reduction(reduce_map[std::string(reduction)]));
    *result = new at::Tensor(output);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Linear(Tensor input, Tensor weight, Tensor bias, Tensor *result) {
  try {
    auto out = torch::nn::functional::linear(*input, *weight,
                                             (bias ? *bias : torch::Tensor()));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *MaxPool2d(Tensor input, int64_t *kernel_data, int64_t kernel_len,
                      int64_t *stride_data, int64_t stride_len,
                      int64_t *padding_data, int64_t padding_len,
                      int64_t *dilation_data, int64_t dilation_len,
                      int8_t ceil_mode, Tensor *result) {
  try {
    auto out = torch::nn::functional::max_pool2d(
        *input, torch::nn::functional::MaxPool2dFuncOptions(
                    torch::IntArrayRef(kernel_data, kernel_len))
                    .stride(torch::IntArrayRef(stride_data, stride_len))
                    .padding(torch::IntArrayRef(padding_data, padding_len))
                    .dilation(torch::IntArrayRef(dilation_data, dilation_len))
                    .ceil_mode(ceil_mode));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *AdaptiveAvgPool2d(Tensor input, int64_t *output_size_data,
                              int64_t output_size_len, Tensor *result) {
  try {
    auto out = torch::nn::functional::adaptive_avg_pool2d(
        *input, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(
                    torch::IntArrayRef(output_size_data, output_size_len)));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
