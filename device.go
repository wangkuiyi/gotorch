package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
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
	// The goroutine scheduler has the following properties that might create
	// new OS threads for Cgo applications:
	// 1. The Go scheduler creates one or more P's, each represents a run queue,
	//    at the program startup time.
	// 2. The Go scheduler creates an OS thread M1 to run goroutines waiting in
	//    the run queue of a P.
	// 3. If a goroutine G makes a Cgo or system call, and the call takes a long
	//    time (>20ms), the Go scheduler would be afraid that it will go on
	//    occupying that thread for more time, so it creates a new OS thread M2
	//    for P to run other goroutines in the run queue.
	// 4. After the Cgo call completes, the goroutine G goes back to the run
	//    queue of the P. The next time it gets a turn to run, it runs on M2
	//    instead of reusing M1, because M2 is now the default thread of P.
	// 5. If G makes a new long-run Cgo call again, the Go scheduler would let G
	//    keeps using M2, and creates M3 for P to run other goroutines in its
	//    run queue.
	// In a nutshell, the above Go scheduler mechanism will create new threads
	// for a Cgo program like GoTorch and cause the main goroutine to migrate
	// from one OS thread to another.

	// The process may create several new threads. Moreover, libtorch and its
	// underlying libraries use thread local storage extensively, both for
	// caching and threads controlling, each thread that calls a `forward`
	// method may create its own computation threads and local storage. The
	// above Go mechanism may create so many threads that the RAM cannot afford,
	// because the Go runtime will never recycle OS threads at the moment. As a
	// result, a GoTorch program will occupy much RAM (the TLSs) and create many
	// threads (the computation threads). (This is an inherent problem of
	// libtorch, it also exists in C++. For a C++ example, See
	// https://github.com/wangkuiyi/gotorch/issues/331)
	// In order to alleviate the problem, we have to limit the default threads
	// number of OMP in GoTorch and lock the main goroutine to a fixed OS
	// thread to avoid migration.

	//
	// Avoid creating too many threads: the original default setting of
	// OMP_NUM_THREADS (defaults to the core number in libtorch) may degrade
	// performance on GPUs because too many threads will increase the overhead
	// of context switching. See https://github.com/wangkuiyi/gotorch/issues/321
	// for details.
	if os.Getenv("OMP_NUM_THREADS") == "" && os.Getenv("MKL_NUM_THREADS") == "" {
		SetNumThreads(int32(runtime.NumCPU()) / 2)
	}

	// Prevent Cgo call from migrating to another system thread, hence the TLS
	// cache in libtorch would not take too much RAM.
	// See https://github.com/wangkuiyi/gotorch/issues/273 for details.
	runtime.LockOSThread()
}
