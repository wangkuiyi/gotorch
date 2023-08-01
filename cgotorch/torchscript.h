#pragma once

#include "cgotorch/torchdef.h"

#ifdef __cplusplus
#include "torch/script.h"

extern "C" {
typedef torch::jit::script::Module *Module;
#else
typedef void *Module;
#endif

const char* loadModule(const char *modelPath, Module *result);
const char* forwardModule(Module module, Tensor input, IValue *output);
void Module_Close(Module a);

#ifdef __cplusplus
}
#endif
