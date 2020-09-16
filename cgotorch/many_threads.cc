// Copyright 2020, GoTorch Authors
#include <stdlib.h>
#include <chrono>  // NOLINT
#include <iostream>
#include <sstream>
#include <string>
#include <thread>  // NOLINT

#include "torch/torch.h"

namespace nn = torch::nn;              // for the literal `ms`
using namespace std::chrono_literals;  // for 10ms // NOLINT

std::mutex mu;

int main(int argc, char* argv[]) {
  std::string argv0 = argv[0];
  auto pos = argv0.rfind('/');
  if (pos != std::string::npos) {
    argv0 = argv0.substr(pos + 1);
  }
  std::stringstream thread_count_command;
  thread_count_command << "ps -T|grep " << argv0 << "| wc -l";
  std::cout << "Thread count command: " << thread_count_command.str()
            << std::endl;
  std::cout << std::string(20, '-') << std::endl;

  std::vector<std::thread> pool;
  auto model = nn::Conv2d(nn::Conv2dOptions(3, 64, 1).stride(1).bias(false));

  auto total = std::thread::hardware_concurrency();
  if (argc > 1) total = std::atoi(argv[1]);

  for (int i = 0; i < total; ++i) {
    pool.push_back(std::thread([&, i] {
      int step = 0;
      while (true) {
        step += 1;
        {
          std::lock_guard<std::mutex> lock(mu);
          std::cout << "Thread " << i << "(" << std::this_thread::get_id()
                    << "), step " << step << std::endl;
          std::cout << "#Threads before `forward`:" << std::endl;
          auto _ = system(thread_count_command.str().c_str());
          std::vector<torch::Tensor> data;
          while (data.size() < 32) data.push_back(torch::rand({3, 599, 599}));
          auto output = model->forward(torch::stack(data));
          std::cout << "#Threads after `forward`:" << std::endl;
          _ = system(thread_count_command.str().c_str());
          std::cout << std::string(20, '-') << std::endl;
        }
        std::this_thread::sleep_for(10ms);  // Yield to another thread
      }
    }));
  }
  for (auto& t : pool) t.join();
}
