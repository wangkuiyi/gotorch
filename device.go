package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/torch_xla/lib -Wl,-rpath ${SRCDIR}/cgotorch/torch_xla/lib -lptxla -lxla_computation_client
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

// IsCUDAAvailable returns true if CUDA is availabl
func IsCUDAAvailable() bool {
	if C.IsCUDAAvailable() == C.bool(true) {
		return true
	}
	return false
}
