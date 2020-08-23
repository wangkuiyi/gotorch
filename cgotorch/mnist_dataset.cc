// Copyright 2020, GoTorch Authors
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

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
