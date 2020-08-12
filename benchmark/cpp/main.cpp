/* Copyright 2020, GoTorch Authors */
#include <torch/torch.h>

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// Where to find the MNIST dataset.
const char* kDataRoot = "/Users/yancey/.cache/mnist";

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

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type = torch::kCPU;
  torch::Device device(torch::kCPU);

  Net model;
  model.to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  std::cout << train_dataset_size << std::endl;
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));

  model.train();
  float loss_value = 0.0;
  int total_throughtput = 0;
  for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch) {
    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();
    size_t batch_idx = 0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(device), targets = batch.target.to(device);
      optimizer.zero_grad();
      auto output = model.forward(data);
      auto loss = torch::nll_loss(output, targets);
      loss_value = loss.template item<float>();
      loss.backward();
      optimizer.step();
      if (batch_idx % 200 == 0) {
        std::printf("Train Epoch: %ld, Batch: %zu, Loss: %.4f\n", epoch,
                    batch_idx, loss_value);
      }
      batch_idx++;
    }
    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    int throughput = (train_dataset_size * 1.0 / time_span.count());
    total_throughtput += throughput;
    std::printf("End Train Epoch: %ld, Throughput: %d sampels/sec\n", epoch,
                throughput);
  }
  std::printf("The average throughtput: %ld sampels/sec\n",
              total_throughtput / kNumberOfEpochs);
}
