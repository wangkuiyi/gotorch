#include "cgotorch/torchscript.h"
#include <iostream>


const char* loadModule(const char *modelPath, Module *result) {
    try {
        *result = new torch::jit::script::Module(torch::jit::load(modelPath));
        return nullptr;
      } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char* forwardModule(Module module, Tensor input, IValue *output) {
    try {
        c10::IValue forwarded = module->forward({*input});
        *output = new c10::IValue(forwarded);
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

void Module_Close(Module a) { delete a; }