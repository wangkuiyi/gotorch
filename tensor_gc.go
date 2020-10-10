package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

var (
	tensorFinalizersWG = &sync.WaitGroup{}
)

// GoTorch should allow goroutines other than the main goroutine to create
// tensors whose lifecycle last for more than one iteration, for example, we
// have to cache tensors in data loading goroutines because they run faster.

// To meet this goal, we have to make sure a `torch.GC()` function call in the
// main goroutine know which tensors are created by the main goroutine, and only
// wait for tensors created in the main goroutine to be freed.

// Actually, we need a "goroutine local storage" mechanism to distinguish the
// main goroutine and the data loading goroutine. However, Go doesn't provide
// an official "goroutine local storage", and the official `context` package
// will impose additional parameters to user API, thus make the API harder to
// use.

// Recall that we've already locked the main goroutine to a fixed OS thread in
// the `init` function in device.go, we can use a C++ `thread_local` to solve
// the problem.

// The three functions `C.GCPrepared()`, `C.PrepareGC()`, and `C.FinishGC()` are
// defined in cgotorch/memory.cc, they use the C++ variable
// `thread_local bool gcPrepared`
// to help control the behavior of the function `torch.GC()` as described above.

// Known limitation:
// 1. `torch.GC()` should be called in only one goroutine in a GoTorch program
//    (typically the main goroutine) exactly before the training/testing loop
//    starts. If we call `torch.GC()` in a unit test case, we have to call
//    `runtime.LockOSThread` manually because the `go test` cmd tool will start
//    new goroutines to run the cases.
// 2. `torch.FinishGC()` should only be called in the same goroutine as the
//    `torch.GC()` function exactly after a training/testing loop ends. Or the
//    goroutine may hang forever.

// In general, `torch.GC()` and `torch.FinishGC()` are low-level functions that
// ordinary users don't have to care about. GoTorch provides high-level APIs
// that wraps the two functions.

// SetTensorFinalizer sets a finalizer to the tensor
func SetTensorFinalizer(t *unsafe.Pointer) {
	// We don't want the following conditional and the finalizer using
	// different gcPrepared values, so we leverage p and closure here.
	p := C.GCPrepared()
	if p != 0 {
		tensorFinalizersWG.Add(1)
	}
	runtime.SetFinalizer(t, func(ct *unsafe.Pointer) {
		C.Tensor_Close(C.Tensor(*ct))
		if p != 0 {
			tensorFinalizersWG.Done()
		}
	})
}

// FinishGC should be called right after a train/predict loop
func FinishGC() {
	GC()
	C.FinishGC()
}

// GC should be called at the beginning inside a train/predict loop
func GC() {
	runtime.GC()
	if C.GCPrepared() == 0 {
		C.PrepareGC()
		return
	}
	tensorFinalizersWG.Wait()
}
