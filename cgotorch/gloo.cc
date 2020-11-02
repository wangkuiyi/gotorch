// Copyright 2020, GoTorch Authors

#include "cgotorch/gloo.h"
#include <memory>
#include <string>
#include <vector>

const char *Gloo_NewFileStore(const char *path, int64_t num_workers,
                              Store *store) {
  try {
    *store = new c10d::FileStore(std::string(path), num_workers);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Gloo_NewTCPStore(const char *addr, int64_t port,
                             int64_t num_workers, int64_t is_server,
                             Store *store) {
  try {
    *store =
        new c10d::TCPStore(std::string(addr), port, num_workers, is_server);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Gloo_NewProcessGroupGloo(Store store, int64_t rank, int64_t size,
                                     ProcessGroupGloo *pg) {
  try {
    auto d = c10d::ProcessGroupGloo::createDefaultDevice();
    auto opt = c10d::ProcessGroupGloo::Options();
    opt.devices.push_back(d);
    *pg = new c10d::ProcessGroupGloo(
        std::shared_ptr<c10d::Store>(static_cast<c10d::Store *>(store)), rank,
        size, opt);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Gloo_allreduce(ProcessGroupGloo pg, Tensor *tensors,
                           int64_t length) {
  try {
    std::vector<torch::Tensor> ts;
    while (ts.size() < length) {
      ts.push_back(**tensors++);
    }
    auto work = static_cast<c10d::ProcessGroupGloo *>(pg)->allreduce(ts);
    work->wait();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Gloo_allreduce_coalesced(ProcessGroupGloo pg, Tensor *tensors,
                                     int64_t length) {
  try {
    std::vector<torch::Tensor> ts;
    while (ts.size() < length) {
      ts.push_back(**tensors++);
    }
    auto work =
        static_cast<c10d::ProcessGroupGloo *>(pg)->allreduce_coalesced(ts);
    work->wait();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

const char *Gloo_broadcast(ProcessGroupGloo pg, Tensor *tensors,
                           int64_t length) {
  try {
    std::vector<torch::Tensor> ts;
    while (ts.size() < length) {
      ts.push_back(**tensors++);
    }
    auto work = static_cast<c10d::ProcessGroupGloo *>(pg)->broadcast(ts);
    work->wait();
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
