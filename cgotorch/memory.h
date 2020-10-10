/* Copyright 2020, GoTorch Authors */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
//  Memory management for Go garbage collector
////////////////////////////////////////////////////////////////////////////////

uint8_t GCPrepared();
void PrepareGC();
void FinishGC();

#ifdef __cplusplus
}
#endif
