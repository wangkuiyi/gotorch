// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

const char *Tensor_Detach(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->detach());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Item(Tensor a, float *result) {
  try {
    *result = a->item<float>();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Mean(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->mean());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

void Tensor_Print(Tensor a) { std::cout << *a << std::endl; }

void Tensor_Close(Tensor a) { delete a; }

const char *Tensor_Save(Tensor tensor, const char *path) {
  try {
    torch::save(*tensor, std::string(path));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_Dim(Tensor tensor, int64_t *dim) {
  try {
    *dim = tensor->dim();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_Shape(Tensor tensor, int64_t *dims) {
  try {
    int i = 0;
    for (int64_t dim : tensor->sizes()) dims[i++] = dim;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_Dtype(Tensor tensor, int8_t *dtype) {
  try {
    auto t = tensor->scalar_type();
    *dtype = static_cast<int8_t>(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// The caller must free the returned string by calling FreeString.
const char *Tensor_String(Tensor a) {
  std::stringstream ss;
  ss << *a;
  std::string s = ss.str();
  char *r = new char[s.size() + 1];
  snprintf(r, s.size() + 1, "%s", s.c_str());
  return r;
}

// Backward, Gradient
void Tensor_Backward(Tensor a) { a->backward(); }
Tensor Tensor_Grad(Tensor a) { return new at::Tensor(a->grad()); }

const char *Tensor_SetData(Tensor self, Tensor new_data) {
  try {
    self->set_data(*new_data);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
