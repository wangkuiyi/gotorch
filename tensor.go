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

func mustNil(err *C.char) {
	if err != nil {
		msg := C.GoString(err)
		C.FreeString(err)
		panic(msg)
	}
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
	var t C.Tensor
	mustNil(C.RandN(C.int(rows), C.int(cols), C.int(rg), &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// RandNByShape return a tensor with given shape
func RandNByShape(shape []int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	var t C.Tensor
	mustNil(C.RandNByShape(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int64_t(len(shape)), C.int(rg), &t))
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

// Relu returns relu of the tensor
func (a *Tensor) Relu() Tensor {
	var t C.Tensor
	mustNil(C.Relu(*a.T, &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// LeakyRelu returns leaky relu of the tensor according to negativeSlope
func (a *Tensor) LeakyRelu(negativeSlope float64) Tensor {
	var t C.Tensor
	mustNil(C.LeakyRelu(*a.T, C.double(negativeSlope), &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// Tanh returns tanh of the current tensor
func (a Tensor) Tanh() Tensor {
	var t C.Tensor
	mustNil(C.Tanh(*a.T, &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// Sigmoid returns sigmoid of the current tensor
func (a Tensor) Sigmoid() Tensor {
	var t C.Tensor
	mustNil(C.Sigmoid(*a.T, &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
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
	var t C.Tensor
	mustNil(C.MM(*a.T, *b.T, &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// LeakyRelu returns leaky relu of the tensor according to negativeSlope
func LeakyRelu(t Tensor, negativeSlope float64) Tensor {
	return t.LeakyRelu(negativeSlope)
}

// Relu returns relu of the tensor
func Relu(t Tensor) Tensor {
	return t.Relu()
}

// Tanh returns tanh of the current tensor
func Tanh(t Tensor) Tensor {
	return t.Tanh()
}

// Sigmoid returns sigmoid of the current tensor
func Sigmoid(t Tensor) Tensor {
	return t.Sigmoid()
}

// Sum returns the sum of all elements in the input tensor
func Sum(a Tensor) Tensor {
	var t C.Tensor
	mustNil(C.Sum(*a.T, &t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// Conv2d does 2d-convolution
func Conv2d(input Tensor, weight Tensor, bias Tensor, stride []int,
	padding []int, dilation []int, groups int) Tensor {
	var t C.Tensor
	if bias.T == nil {
		mustNil(
			C.Conv2d(
				*input.T,
				*weight.T,
				nil,
				(*C.int64_t)(unsafe.Pointer(&stride[0])),
				C.int64_t(len(stride)),
				(*C.int64_t)(unsafe.Pointer(&padding[0])),
				C.int64_t(len(padding)),
				(*C.int64_t)(unsafe.Pointer(&dilation[0])),
				C.int64_t(len(dilation)),
				C.int64_t(groups),
				&t))
		setTensorFinalizer(&t)
		return Tensor{&t}
	}
	mustNil(
		C.Conv2d(
			*input.T,
			*weight.T,
			*bias.T,
			(*C.int64_t)(unsafe.Pointer(&stride[0])),
			C.int64_t(len(stride)),
			(*C.int64_t)(unsafe.Pointer(&padding[0])),
			C.int64_t(len(padding)),
			(*C.int64_t)(unsafe.Pointer(&dilation[0])),
			C.int64_t(len(dilation)),
			C.int64_t(groups),
			&t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// ConvTranspose2d does 2d-fractionally-strided convolution
func ConvTranspose2d(
	input, weight, bias Tensor,
	stride, padding, outputPadding []int,
	groups int, dilation []int) Tensor {

	var cbias, t C.Tensor
	if bias.T != nil {
		cbias = *bias.T
	}

	mustNil(C.ConvTranspose2d(
		*input.T,
		*weight.T,
		cbias,
		(*C.int64_t)(unsafe.Pointer(&stride[0])),
		C.int64_t(len(stride)),
		(*C.int64_t)(unsafe.Pointer(&padding[0])),
		C.int64_t(len(padding)),
		(*C.int64_t)(unsafe.Pointer(&outputPadding[0])),
		C.int64_t(len(outputPadding)),
		C.int64_t(groups),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])),
		C.int64_t(len(dilation)),
		&t))
	setTensorFinalizer(&t)
	return Tensor{&t}
}
