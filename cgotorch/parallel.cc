// Copyright 2020, GoTorch Authors
#ifdef WITH_CUDA
#include <torch/nn/parallel/data_parallel.h>
#endif

#include <memory>

#include "cgotorch/parallel.h"

typedef Tensor (*ForwardMethod)(void *, Tensor);

// goModule wraps the `goModuleForward` funciton defined in nn/parallel.go into
// a class method
struct goModule : torch::nn::Module {
  char *m_;
  ForwardMethod f_;
  goModule(char *m, void *f) : m_(m), f_(reinterpret_cast<ForwardMethod>(f)) {}
  at::Tensor forward(at::Tensor input) {  // NOLINT: include_what_you_use
    // TODO(shendiaomo): check the return value of `f_`
    return *f_(m_, &input);
  }
};

const char *DataParallel(char *go_module, void *f, Tensor input,
                         Device *devices, int64_t size, Device *output,
                         int64_t dim) {
#ifdef WITH_CUDA
  try {
    if (input == nullptr) {
      throw std::runtime_error(
          "invalid memory address or nil pointer dereference of input tensor");
    }
    torch::nn::parallel::data_parallel(std::make_shared<goModule>(go_module, f),
                                       *input);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
#else
  return exception_str(
      "Parallel API needs -DWITH_CUDA on building libcgotorch.so");
#endif
}
