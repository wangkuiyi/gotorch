#include <omp.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;  // for the literal `s` and `ms`

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
          assert(std::system(thread_count_command.str().c_str()) == 0);

          // The following OMP code is from `at::parallel_for`
          // There's a similar problem in
          // https://github.com/pytorch/pytorch/issues/32008
#pragma omp parallel if (omp_get_max_threads() > 1 && !omp_in_parallel())
          omp_get_num_threads();

          std::cout << "#Threads after `forward`:" << std::endl;
          assert(std::system(thread_count_command.str().c_str()) == 0);
          std::cout << std::string(20, '-') << std::endl;
        }
        std::this_thread::sleep_for(1s);  // Yield to another thread
      }
    }));
  }
  for (auto& t : pool) t.join();
}
