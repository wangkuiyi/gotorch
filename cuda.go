package gotorch

// #cgo CFLAGS: -I ${SRCDIR} -I ${SRCDIR}/cgotorch/libtorch/include
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import "unsafe"

// CUDAStream struct wrapped Nvidia CUDA Stream
type CUDAStream struct {
	P C.CUDAStream
}

// Query returns true if all tasks completed on this CUDA stream
func (s CUDAStream) Query() bool {
	var b int8
	MustNil(unsafe.Pointer(C.CUDA_Query(s.P, (*C.int8_t)(&b))))
	return b != 0
}

// Synchronize wait until all tasks completed on this CUDA stream
func (s CUDAStream) Synchronize() {
	MustNil(unsafe.Pointer(C.CUDA_Synchronize(s.P)))
}

// GetCurrentCUDAStream returns the current stream on device
func GetCurrentCUDAStream(device Device) CUDAStream {
	var stream C.CUDAStream
	MustNil(unsafe.Pointer(C.CUDA_GetCurrentCUDAStream(&stream, &device.T)))
	return CUDAStream{stream}
}

// SetCurrentCUDAStream set stream as the current CUDA stream
func SetCurrentCUDAStream(stream CUDAStream) {
	MustNil(unsafe.Pointer(C.CUDA_SetCurrentCUDAStream(stream.P)))
}

// NewCUDAStream returns a new CUDA stream from the pool
func NewCUDAStream(device Device) CUDAStream {
	var stream C.CUDAStream
	MustNil(unsafe.Pointer(C.CUDA_GetCUDAStreamFromPool(&stream, &device.T)))
	return CUDAStream{stream}
}
