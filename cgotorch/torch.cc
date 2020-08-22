// Copyright 2020, GoTorch Authors
#include "torch/torch.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const char *e) {
  auto len = strlen(e);
  auto r = new char[len + 1];
  snprintf(r, len + 1, "%s", e);
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations
////////////////////////////////////////////////////////////////////////////////

const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::randn(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Rand(int64_t *size, int64_t length, int64_t require_grad,
                 Tensor *result) {
  try {
    at::Tensor t = torch::rand(torch::IntArrayRef(size, length),
                               at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::empty(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Equal(Tensor a, Tensor b, int64_t *result) {
  try {
    *result = at::equal(*a, *b) ? 1 : 0;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c = at::mm(*a, *b);
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Sum(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *SumByDim(Tensor a, int64_t dim, int8_t keepDim, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum(dim, keepDim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Relu(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->relu());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result) {
  try {
    *result = new at::Tensor(at::leaky_relu(*a, negative_slope));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Tanh(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->tanh());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Sigmoid(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sigmoid());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Add(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(torch::add(*a, *other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Add_(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(a->add_(*other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Flatten(Tensor a, int64_t startDim, int64_t endDim,
                    Tensor *result) {
  try {
    *result = new at::Tensor(torch::flatten(*a, startDim, endDim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *TopK(Tensor a, int64_t k, int64_t dim, int8_t largest,
                 int8_t sorted, Tensor *values, Tensor *indices) {
  try {
    auto outputs = torch::topk(*a, k, dim, largest, sorted);
    *values = new at::Tensor(std::get<0>(outputs));
    *indices = new at::Tensor(std::get<1>(outputs));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor *result) {
  try {
    *result = new at::Tensor(torch::transpose(*a, dim0, dim1));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *ExpandAs(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->expand_as(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Eq(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::eq(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor *result) {
  try {
    *result = new at::Tensor(torch::index_select(*a, dim, *index));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len) {
  try {
    *result = new at::Tensor(a->view(torch::IntArrayRef(size, size_len)));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->log_softmax(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Squeeze(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->squeeze());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *SqueezeWithDim(Tensor a, int64_t dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->squeeze(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

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

void FreeString(const char *s) { delete[] s; }

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
////////////////////////////////////////////////////////////////////////////////
//  Dataset, DataLoader, and Iterator torch.utils.data
////////////////////////////////////////////////////////////////////////////////

const char *Dataset_MNIST(const char *data_root, Dataset *dataset) {
  try {
    dataset->p = new torch::data::datasets::MNIST(std::string(data_root));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

void MNIST_Close(Dataset d) { delete d.p; }

void Dataset_Normalize(Dataset *dataset, double mean, double stddev) {
  dataset->normalize = new torch::data::transforms::Normalize<>(mean, stddev);
}

using TypeMapDataset = torch::data::datasets::MapDataset<
    torch::data::datasets::MapDataset<torch::data::datasets::MNIST,
                                      torch::data::transforms::Normalize<>>,
    torch::data::transforms::Stack<torch::data::Example<>>>;
using TypeDataLoader =
    torch::data::StatelessDataLoader<TypeMapDataset,
                                     torch::data::samplers::SequentialSampler>;

using TypeIterator = torch::data::Iterator<TypeDataLoader::BatchType>;
DataLoader MakeDataLoader(Dataset dataset, int64_t batchsize) {
  auto map_dataset = dataset.p->map(*dataset.normalize)
                         .map(torch::data::transforms::Stack<>());
  auto p =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          map_dataset, batchsize);
  return p.release();
}

Iterator Loader_Begin(DataLoader loader) {
  return new TypeIterator(static_cast<TypeDataLoader *>(loader)->begin());
}

void Iterator_Batch(Iterator iter, Tensor *data, Tensor *target) {
  auto i = *static_cast<TypeIterator *>(iter);
  *data = new at::Tensor(i->data);
  *target = new at::Tensor(i->target);
}

bool Loader_Next(DataLoader loader, Iterator iter) {
  return ++*static_cast<TypeIterator *>(iter) !=
         static_cast<TypeDataLoader *>(loader)->end();
}

void Loader_Close(DataLoader loader) {
  delete static_cast<TypeDataLoader *>(loader);
}

////////////////////////////////////////////////////////////////////////////////
// Device
////////////////////////////////////////////////////////////////////////////////
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
