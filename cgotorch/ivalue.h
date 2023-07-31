#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
extern "C" {
#endif

const bool IValue_isTensor(const IValue ivalue);
const bool IValue_isTuple(const IValue ivalue);
const char* IValue_toTuple(const IValue ivalue, IValue **output, int *length);
const char* IValue_toTensor(const IValue ivalue, Tensor *output);

#ifdef __cplusplus
}
#endif
