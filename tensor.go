package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

import (
	"runtime"
	"sync"
)

var cTensorFinalizersWG = &sync.WaitGroup{}

func setTensorFinalizer(t *C.Tensor) {
	cTensorFinalizersWG.Add(1)
	runtime.SetFinalizer(t, func(t *C.Tensor) {
		go func() {
			C.Tensor_Close(*t)
			cTensorFinalizersWG.Done()
		}()
	})
}

func GC() {
	runtime.GC()
	cTensorFinalizersWG.Wait()
}

// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T *C.Tensor
}

// String returns the Tensor as a string
func (a Tensor) String() string {
	s := C.Tensor_String(*a.T)
	r := C.GoString(s)
	C.FreeString(s)
	return r
}

// Print the tensor
func (a Tensor) Print() {
	C.Tensor_Print(*a.T)
}

// Backward compute the gradient of current tensor
func (a Tensor) Backward() {
	C.Tensor_Backward(*a.T)
}

// Grad returns a reference of the gradient
func (a Tensor) Grad() Tensor {
	t := C.Tensor_Grad(*a.T)
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// MM multiplies each element of the input two tensors
func MM(a, b Tensor) Tensor {
	t := C.MM(a.T, b.T)
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// Sum returns the sum of all elements in the input tensor
func Sum(a Tensor) Tensor {
	t := C.Sum(a.T)
	setTensorFinalizer(&t)
	return Tensor{&t}
}
