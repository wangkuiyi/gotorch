// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

const char *CreateMNISTDataset(const char *data_root, MNISTDataset *dataset) {
  try {
    dataset->p = new torch::data::datasets::MNIST(std::string(data_root));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

void MNISTDataset_Close(MNISTDataset d) { delete d.p; }

void MNISTDataset_Normalize(MNISTDataset *dataset, double *mean,
                            int64_t mean_len, double *stddev,
                            int64_t stddev_len) {
  dataset->normalize = new torch::data::transforms::Normalize<>(
      at::ArrayRef<double>(mean, mean_len),
      at::ArrayRef<double>(stddev, stddev_len));
}

namespace detail {
using Dataset = torch::data::datasets::MapDataset<
    torch::data::datasets::MapDataset<torch::data::datasets::MNIST,
                                      torch::data::transforms::Normalize<>>,
    torch::data::transforms::Stack<torch::data::Example<>>>;
using Loader =
    torch::data::StatelessDataLoader<Dataset,
                                     torch::data::samplers::SequentialSampler>;
using Iterator = torch::data::Iterator<Loader::BatchType>;
}  // namespace detail

MNISTLoader CreateMNISTLoader(MNISTDataset dataset, int64_t batchsize) {
  auto map_dataset = dataset.p->map(*dataset.normalize)
                         .map(torch::data::transforms::Stack<>());
  auto p =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          map_dataset, batchsize);
  return p.release();
}

void MNISTLoader_Close(MNISTLoader loader) {
  delete static_cast<detail::Loader *>(loader);
}

MNISTIterator MNISTLoader_Begin(MNISTLoader loader) {
  return new detail::Iterator(static_cast<detail::Loader *>(loader)->begin());
}

void MNISTIterator_Batch(MNISTIterator iter, Tensor *data, Tensor *target) {
  auto i = *static_cast<detail::Iterator *>(iter);
  *data = new at::Tensor(i->data);
  *target = new at::Tensor(i->target);
}

bool MNISTIterator_Next(MNISTIterator iter, MNISTLoader loader) {
  return ++*static_cast<detail::Iterator *>(iter) !=
         static_cast<detail::Loader *>(loader)->end();
}

bool MNISTIterator_IsEnd(MNISTIterator iter, MNISTLoader loader) {
  return *static_cast<detail::Iterator *>(iter) ==
         static_cast<detail::Loader *>(loader)->end();
}

void MNISTIterator_Close(MNISTIterator iter) {
  delete static_cast<detail::Iterator *>(iter);
}
