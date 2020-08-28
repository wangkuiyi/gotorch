// Copyright 2020. GoTorch Authors.
#include <openssl/md5.h>

#include <iostream>

#include "torch/csrc/api/include/torch/serialize.h"
#include "torch/script.h"

int main() {
  auto a = torch::eye(3);
  std::cout << "Generated tensor = " << a << std::endl;

  if (!a.is_cuda()) {
    a = a.cpu();
  }

  auto e = torch::pickle_save(a);
  std::cout << "Encoded buffer size = " << e.size() << std::endl;

  unsigned char checksum[MD5_DIGEST_LENGTH];
  MD5(reinterpret_cast<unsigned char*>(e.data()), e.size(), checksum);
  for (int c : checksum) {
    std::cout << std::hex << std::setw(2) << std::setfill('0') << c;
  }
  std::cout << std::endl;

  auto v = torch::pickle_load(e);
  std::cout << "Loaded tensor = " << v << std::endl;
  return 0;
}
