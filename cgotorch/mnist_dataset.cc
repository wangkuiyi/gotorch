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

void MNISTDataset_Normalize(MNISTDataset *dataset, double mean, double stddev) {
  dataset->normalize = new torch::data::transforms::Normalize<>(mean, stddev);
}

using MNISTMapDataset = torch::data::datasets::MapDataset<
    torch::data::datasets::MapDataset<torch::data::datasets::MNIST,
                                      torch::data::transforms::Normalize<>>,
    torch::data::transforms::Stack<torch::data::Example<>>>;

using MNISTDataLoader =
    torch::data::StatelessDataLoader<MNISTMapDataset,
                                     torch::data::samplers::SequentialSampler>;

using MNISTDataLoaderIterator =
    torch::data::Iterator<MNISTDataLoader::BatchType>;

MNISTLoader CreateMNISTLoader(MNISTDataset dataset, int64_t batchsize) {
  auto map_dataset = dataset.p->map(*dataset.normalize)
                         .map(torch::data::transforms::Stack<>());
  auto p =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          map_dataset, batchsize);
  return p.release();
}

void MNISTLoader_Close(MNISTLoader loader) {
  delete static_cast<MNISTDataLoader *>(loader);
}

MNISTIterator MNISTLoader_Begin(MNISTDataLoader loader) {
  return new MNISTDataLoaderIterator(
      static_cast<MNISTDataLoader *>(loader)->begin());
}

void MNISTIterator_Batch(MNISTIterator iter, Tensor *data, Tensor *target) {
  auto i = *static_cast<MNISTDataLoaderIterator *>(iter);
  *data = new at::Tensor(i->data);
  *target = new at::Tensor(i->target);
}

bool MNISTIterator_Next(Iterator iter, MNISTLoader loader) {
  return ++*static_cast<MNISTDataLoaderIterator *>(iter) !=
         static_cast<MNISTDataLoader *>(loader)->end();
}
