package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

var (
	tensorFinalizersWG = &sync.WaitGroup{}
	gcPrepared         = false
)

// SetTensorFinalizer sets a finalizer to the tensor
func SetTensorFinalizer(t *unsafe.Pointer) {
	// We don't want the following conditional and the finalizer using
	// different gcPrepared values, so we leverage p and closure here.
	p := gcPrepared
	if p {
		tensorFinalizersWG.Add(1)
	}
	runtime.SetFinalizer(t, func(ct *unsafe.Pointer) {
		C.Tensor_Close(C.Tensor(*ct))
		if p {
			tensorFinalizersWG.Done()
		}
	})
}

// FinishGC should be called right after a train/predict loop
func FinishGC() {
	GC()
	gcPrepared = false
}

// GC should be called at the beginning inside a train/predict loop
func GC() {
	runtime.GC()
	if !gcPrepared {
		gcPrepared = true
		return
	}
	tensorFinalizersWG.Wait()
}
