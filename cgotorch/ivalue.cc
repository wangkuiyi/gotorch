#include "cgotorch/ivalue.h"
#include <iostream>


const bool IValue_isTensor(const IValue ivalue) {
    return ivalue->isTensor();
}

const bool IValue_isTuple(const IValue ivalue) {
    return ivalue->isTuple();
}

const char* IValue_toTuple(const IValue ivalue, IValue **output, int *length) {
    try {
        auto elements = ivalue->toTuple()->elements();
        *length = elements.size();
        *output = new IValue[*length];
        for (int i = 0; i < *length; i++) {
            (*output)[i] = new c10::IValue(elements[i]);
        }
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

const char* IValue_toTensor(const IValue ivalue, Tensor *output) {
    try {
        *output = new at::Tensor(ivalue->toTensor());
        return nullptr;
    } catch (const std::exception &e) {
        return exception_str(e.what());
    }
}

void IValue_Close(IValue ivalue) { delete ivalue; }
