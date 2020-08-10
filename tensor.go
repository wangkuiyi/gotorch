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
	runtime.SetFinalizer(t, func(ct *C.Tensor) {
		go func() {
			C.Tensor_Close(*ct)
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

// MustNil asserts error to be nil
func MustNil(err unsafe.Pointer) {
	if err != nil {
		msg := C.GoString((*C.char)(err))
		C.FreeString((*C.char)(err))
		panic(msg)
	}
}

// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T unsafe.Pointer
}

// RandN returns a tensor filled with standard normal distribution, torch.randn
func RandN(shape []int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.RandN((*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int64_t(len(shape)), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{unsafe.Pointer(t)}
}

// Empty returns a tensor filled with random number, torch.empty
func Empty(shape []int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(
		unsafe.Pointer(C.Empty((*C.int64_t)(unsafe.Pointer(&shape[0])),
			C.int64_t(len(shape)), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	SetTensorFinalizer(unsafe.Pointer(t))
	return Tensor{unsafe.Pointer(t)}
}

// Zeros initialization, torch.nn.init.zeros_
func Zeros(a *Tensor) {
	MustNil(unsafe.Pointer(C.Zeros_((*C.Tensor)(&a.T))))
}

// Uniform initialization, torch.nn.init.uniform_
func Uniform(a *Tensor, low, high float64) {
	MustNil(unsafe.Pointer(C.Uniform_((*C.Tensor)(&a.T), C.double(low), C.double(high))))
}

// KaimingUniform initialization, torch.nn.init.kaiming_uniform_
func KaimingUniform(input *Tensor, a float64, fanMode string,
	nonLinearity string) {
	MustNil(unsafe.Pointer(C.KaimingUniform_(C.double(a), C.CString(fanMode),
		C.CString(nonLinearity), (*C.Tensor)(&input.T))))
}

// CalculateFanInAndFanOut torch.nn.init._calculate_fan_in_and_fan_out
func CalculateFanInAndFanOut(input Tensor) (int, int) {
	var fanIn, fanOut int
	MustNil(unsafe.Pointer(C.CalculateFanInAndFanOut(
		C.Tensor(input.T),
		(*C.int64_t)(unsafe.Pointer(&fanIn)),
		(*C.int64_t)(unsafe.Pointer(&fanOut)))))
	return fanIn, fanOut
}

// String returns the Tensor as a string
func (a Tensor) String() string {
	s := C.Tensor_String(C.Tensor(a.T))
	r := C.GoString(s)
	C.FreeString(s)
	return r
}

// Print the tensor
func (a Tensor) Print() {
	C.Tensor_Print(C.Tensor(a.T))
}

// Close the tensor
func (a *Tensor) Close() {
	if a.T != nil {
		C.Tensor_Close(C.Tensor(a.T))
		a.T = nil
	}
}

// Relu returns relu of the tensor
func (a *Tensor) Relu() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Relu(C.Tensor(a.T), &t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// LeakyRelu returns leaky relu of the tensor according to negativeSlope
func (a *Tensor) LeakyRelu(negativeSlope float64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.LeakyRelu(C.Tensor(a.T), C.double(negativeSlope), &t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// Tanh returns tanh of the current tensor
func (a Tensor) Tanh() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tanh(C.Tensor(a.T), &t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// Sigmoid returns sigmoid of the current tensor
func (a Tensor) Sigmoid() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tanh(C.Tensor(a.T), &t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// Backward compute the gradient of current tensor
func (a Tensor) Backward() {
	C.Tensor_Backward(C.Tensor(a.T))
}

// Grad returns a reference of the gradient
func (a Tensor) Grad() Tensor {
	t := C.Tensor_Grad(C.Tensor(a.T))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// MM multiplies each element of the input two tensors
func MM(a, b Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.MM(C.Tensor(a.T), C.Tensor(b.T), &t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
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
	MustNil(unsafe.Pointer(C.Sum(C.Tensor(a.T), &t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// BatchNorm does batch nomalization for `input`
func BatchNorm(input, weight, bias, runningMean, runningVar Tensor,
	training bool, momentum, eps float64, cudnnEnabled bool) Tensor {
	var cTraining, cCudnnEnabled C.int8_t
	if training {
		cTraining = 1
	}
	if cudnnEnabled {
		cCudnnEnabled = 1

	}
	var t C.Tensor
	MustNil(
		unsafe.Pointer(C.BatchNorm(
			C.Tensor(input.T),
			C.Tensor(weight.T),
			(*C.Tensor)(&bias.T),
			(*C.Tensor)(&runningMean.T),
			(*C.Tensor)(&runningVar.T),
			cTraining,
			C.double(momentum),
			C.double(eps),
			cCudnnEnabled,
			&t)))
	SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}
