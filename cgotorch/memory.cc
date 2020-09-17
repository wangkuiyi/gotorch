// Copyright 2020, GoTorch Authors
#include "torch/torch.h"

// FIXME(shendiaomo): including cgotorch.h before torch/torch.h will fail
#include "cgotorch/cgotorch.h"

thread_local bool gcPrepared;

uint8_t GCPrepared() { return gcPrepared; }

void PrepareGC() { gcPrepared = true; }

void FinishGC() { gcPrepared = false; }
