#include "torch/torch.h"
#include "cgotorch/cgotorch.h"


thread_local bool gcPrepared;

uint8_t GCPrepared() { return gcPrepared; }

void PrepareGC() { gcPrepared = true; }

void FinishGC() { gcPrepared = false; }
