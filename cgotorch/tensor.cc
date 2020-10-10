// Copyright 2020, GoTorch Authors
#include "cgotorch/tensor.h"

#include <string>
#include <vector>

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

const char *ItemInt64(Tensor a, int64_t *result) {
  try {
    *result = a->item<int64_t>();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *ItemFloat64(Tensor a, double *result) {
  try {
    *result = a->item<double>();
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

const char *Tensor_Load(const char *path, Tensor *tensor) {
  try {
    *tensor = new at::Tensor();
    torch::load(**tensor, path);
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

void FreeString(const char *s) { delete[] s; }

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

const char *Tensor_CUDA(Tensor input, Device device, int8_t non_blocking,
                        Tensor *output) {
  try {
    if (!device->is_cuda()) {
      return exception_str("the device should be CUDA device");
    }
    auto result = input->to(*device, static_cast<bool>(non_blocking));
    *output = new at::Tensor(result);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_CastTo(Tensor input, int8_t dtype, Tensor *output) {
  try {
    auto result = input->to(static_cast<at::ScalarType>(dtype));
    *output = new at::Tensor(result);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_CopyTo(Tensor input, Device device, Tensor *output) {
  try {
    auto result = input->to(*device);
    *output = new at::Tensor(result);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_PinMemory(Tensor input, Tensor *output) {
  try {
    auto result = input->pin_memory();
    *output = new at::Tensor(result);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
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

// from_blob does not allocate a new space, it returns a C++ tensor view
// of a Go array. When the array in Go world is freed, the tensor in C++
// world becomes illegal. We must switch to use deep copy.
const char *Tensor_FromBlob(void *data, int8_t dtype, int64_t *sizes_data,
                            int64_t sizes_data_len, Tensor *result) {
  try {
    auto t = at::from_blob(data, at::IntArrayRef(sizes_data, sizes_data_len),
                           torch::dtype(at::ScalarType(dtype)))
                 .clone();
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_Index(Tensor input, int64_t *index, int64_t index_len,
                         Tensor *result) {
  try {
    std::vector<at::indexing::TensorIndex> indices;
    for (int i = 0; i < static_cast<int>(index_len); i++) {
      indices.push_back(at::indexing::TensorIndex(index[i]));
    }
    at::ArrayRef<at::indexing::TensorIndex> ref(indices.data(), index_len);
    *result = new at::Tensor(input->index(ref));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
