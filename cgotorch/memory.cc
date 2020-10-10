// Copyright 2020, GoTorch Authors
#include "cgotorch/memory.h"

thread_local bool gcPrepared = false;

uint8_t GCPrepared() { return gcPrepared; }

void PrepareGC() { gcPrepared = true; }

void FinishGC() { gcPrepared = false; }
