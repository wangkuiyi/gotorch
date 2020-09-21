package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"os"
	"runtime"
	"unsafe"
)

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

// IsCUDNNAvailable returns true if cuDNN is available
func IsCUDNNAvailable() bool {
	return C.IsCUDNNAvailable() == C.bool(true)
}

// SetNumThreads sets the number of threads created by GoTorch
func SetNumThreads(n int32) {
	C.SetNumThreads(C.int32_t(n))
}

func init() {
	// Avoid creating too many threads: the original default setting of OMP_NUM_THREADS
	// (defaults to the core number) may degrade performance on GPUs.
	if os.Getenv("OMP_NUM_THREADS") == "" && os.Getenv("MKL_NUM_THREADS") == "" {
		SetNumThreads(int32(runtime.NumCPU()) / 2)
	}
	// Prevent Cgo call from migrating to another system thread, hence the TLS cache in
	// libtorch would not take too much RAM. See https://github.com/wangkuiyi/gotorch/issues/273
	// for details.
	runtime.LockOSThread()
}
