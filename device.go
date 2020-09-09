package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import "unsafe"

// Device wrapper a pointer to C.Device
type Device struct {
	T C.Device
}

// NewDevice returns a Device
func NewDevice(deviceType string) Device {
	var t C.Device
	MustNil(unsafe.Pointer(C.Torch_Device(C.CString(deviceType), &t)))
	return Device{t}
}

// IsCUDAAvailable returns true if CUDA is available
func IsCUDAAvailable() bool {
	return C.IsCUDAAvailable() == C.bool(true)
}

// IscuDNNAvailable returns true if cuDNN is available
func IscuDNNAvailable() bool {
	return C.IsCUDNNAvailable() == C.bool(true)
}
