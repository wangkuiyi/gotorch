// Copyright 2020, GoTorch Authors
#include "cgotorch/cgotorch.h"
#include "torch/torch.h"

thread_local bool gcPrepared;

uint8_t GCPrepared() { return gcPrepared; }

void PrepareGC() { gcPrepared = true; }

void FinishGC() { gcPrepared = false; }
