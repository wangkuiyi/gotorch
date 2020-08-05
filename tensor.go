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

func setTensorFinalizer(t *C.Tensor) {
	// We don't want the following conditional and the finalizer using
	// different gcPrepared values, so we leverage p and closure here.
	p := gcPrepared
	if p {
		tensorFinalizersWG.Add(1)
	}
	runtime.SetFinalizer(t, func(t *C.Tensor) {
		go func() {
			C.Tensor_Close(*t)
			if p {
				tensorFinalizersWG.Done()
			}
		}()
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

// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T *C.Tensor
}

// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	t := C.RandN(C.int(rows), C.int(cols), C.int(rg))
	setTensorFinalizer(&t)
	return Tensor{&t}
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

// Close the tensor
func (a *Tensor) Close() {
	if a.T != nil {
		C.Tensor_Close(*a.T)
		a.T = nil
	}
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
	t := C.MM(*a.T, *b.T)
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// Sum returns the sum of all elements in the input tensor
func Sum(a Tensor) Tensor {
	t := C.Sum(*a.T)
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// Conv2d does 2d-convolution
func Conv2d(input Tensor, weight Tensor, bias Tensor, stride []int,
	padding []int, dilation []int, groups int) Tensor {
	if bias.T == nil {
		t := C.Conv2d(
			*input.T,
			*weight.T,
			nil,
			(*C.longlong)(unsafe.Pointer(&stride[0])),
			C.longlong(len(stride)),
			(*C.longlong)(unsafe.Pointer(&padding[0])),
			C.longlong(len(padding)),
			(*C.longlong)(unsafe.Pointer(&dilation[0])),
			C.longlong(len(dilation)),
			C.longlong(groups),
		)
		setTensorFinalizer(&t)
		return Tensor{&t}
	}
	t := C.Conv2d(
		*input.T,
		*weight.T,
		*bias.T,
		(*C.longlong)(unsafe.Pointer(&stride[0])),
		C.longlong(len(stride)),
		(*C.longlong)(unsafe.Pointer(&padding[0])),
		C.longlong(len(padding)),
		(*C.longlong)(unsafe.Pointer(&dilation[0])),
		C.longlong(len(dilation)),
		C.longlong(groups),
	)
	setTensorFinalizer(&t)
	return Tensor{&t}
}
