/* Copyright 2020, GoTorch Authors */
#include <torch/torch.h>
#include <unistd.h>

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 5;

struct Net : torch::nn::Module {
  Net() : fc1(28 * 28, 512), fc2(512, 512), fc3(512, 10) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.view({-1, 28 * 28});
    x = fc1->forward(x);
    x = torch::tanh(x);
    x = fc2->forward(x);
    x = torch::tanh(x);
    x = fc3->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::Linear fc3;
};

void process_mem_usage(double& resident_set) {
  using std::ifstream;
  using std::ios_base;
  using std::string;

  resident_set = 0.0;

  // 'file' stat seems to give the most reliable results
  //
  ifstream stat_stream("/proc/self/stat", ios_base::in);

  // dummy vars for leading entries in stat that we don't care about
  //
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  string O, itrealvalue, starttime;

  // the two fields we want
  //
  int64_t vsize;
  int64_t rss;

  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
      starttime >> vsize >> rss;  // don't care about the rest

  stat_stream.close();

  int64_t page_size_kb = sysconf(_SC_PAGE_SIZE) /
                         1024;  // in case x86-64 is configured to use 2MB pages
  resident_set = rss * page_size_kb / 1024.0 / 1024.0;
}

auto main() -> int {
  torch::manual_seed(1);
  std::string homedir = getenv("HOME");

  torch::DeviceType device_type = torch::kCPU;
  torch::Device device(torch::kCPU);

  Net model;
  model.to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(homedir + "/.cache/mnist")
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));

  model.train();
  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch) {
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(device), targets = batch.target.to(device);
      optimizer.zero_grad();
      auto output = model.forward(data);
      auto loss = torch::nll_loss(output, targets);
      loss.backward();
      optimizer.step();
    }
    double res_usage;
    process_mem_usage(res_usage);
    std::printf("End train epoch: %zu, memory usage: %.4f G\n", epoch,
                res_usage);
  }
  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                start_time);
  float throughput =
      train_dataset_size * kNumberOfEpochs * 1.0 / time_span.count();
  std::printf("The throughput: %.6f samples/sec\n", throughput);
}
