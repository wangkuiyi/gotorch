// Copyright 2020, GoTorch Authors
#include "cgotorch/pickle.h"

#include <vector>

const char* Tensor_Encode(Tensor a, ByteBuffer* r) {
  try {
    *r = new (std::vector<char>);
    if (a->is_cuda() || a->is_hip()) {
      **r = torch::pickle_save(a->cpu());  // Move GPU tensors to CPU.
    } else {
      **r = torch::pickle_save(*a);
    }
    return nullptr;
  } catch (const std::exception& e) {
    return exception_str(e.what());
  }
}

void* ByteBuffer_Data(ByteBuffer buf) { return buf->data(); }

int64_t ByteBuffer_Size(ByteBuffer buf) { return uint64_t(buf->size()); }

void ByteBuffer_Free(ByteBuffer buf) {
  if (buf != nullptr) {
    delete buf;
  }
}

const char* Tensor_Decode(void* addr, int64_t size, Tensor* r) {
  try {
    auto data = static_cast<const char*>(addr);
    std::vector<char> buf(data, data + static_cast<int>(size));
    *r = new at::Tensor();
    **r = torch::pickle_load(buf).toTensor();
    return nullptr;
  } catch (const std::exception& e) {
    return exception_str(e.what());
  }
}
