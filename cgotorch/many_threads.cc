#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "torch/torch.h"

namespace nn = torch::nn;  // for the literal `ms`

using namespace std::chrono_literals;

std::mutex mu;

int main(int argc, char* argv[]) {
  std::string argv0 = argv[0];
  if (auto pos = argv0.rfind('/'); pos != std::string::npos) {
    argv0 = argv0.substr(pos + 1);
  }
  std::stringstream thread_count_command;
  thread_count_command << "ps -T|grep " << argv0 << "| wc -l";
  std::cout << "Thread count command: " << thread_count_command.str()
            << std::endl;
  std::cout << std::string(20, '-') << std::endl;

  std::vector<std::thread> pool;
  // Prepare inputs
  auto x = torch::arange(75, torch::dtype(torch::kFloat).requires_grad(false))
               .reshape({1, 3, 5, 5});
  auto weight =
      torch::arange(54, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({2, 3, 3, 3});
  auto kernel_size = weight.sizes().slice(2);
  auto output = at::empty({0}, x.options());
  auto finput = at::empty({0}, x.options());
  auto fgrad_input = at::empty({0}, x.options());

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
          at::native::slow_conv2d_forward_out_cpu(output, finput, fgrad_input,
                                                  x, weight, kernel_size, {},
                                                  {1, 1}, {0, 0});
          std::cout << "#Threads after `forward`:" << std::endl;
          _ = system(thread_count_command.str().c_str());
          std::cout << std::string(20, '-') << std::endl;
        }
        std::this_thread::sleep_for(2s);  // Yield to another thread
      }
    }));
  }
  for (auto& t : pool) t.join();
}
