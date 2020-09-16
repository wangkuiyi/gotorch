package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch -I ${SRCDIR}/cgotorch/libtorch/include
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import "unsafe"

// Stream struct wrapped CUDAStream
type Stream struct {
	P C.CUDAStream
}

// GetCurrentStream returns the current stream on device
func GetCurrentStream(device Device) Stream {
	var stream C.CUDAStream
	MustNil(unsafe.Pointer(C.CUDA_GetCurrentStream(&stream, &device.T)))
	return Stream{stream}
}
