/* Copyright 2020, GoTorch Authors */
#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// Pickle encode/decode Tensors
////////////////////////////////////////////////////////////////////////////////

const char *Tensor_Encode(Tensor, ByteBuffer *);

void *ByteBuffer_Data(ByteBuffer);
int64_t ByteBuffer_Size(ByteBuffer);
void ByteBuffer_Free(ByteBuffer);

const char *Tensor_Decode(void *addr, int64_t size, Tensor *);

#ifdef __cplusplus
}
#endif
