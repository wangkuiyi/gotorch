package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import "runtime"

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
	ct := C.Tensor_Grad(*a.T)
	return Tensor{&ct}
}

// Close the tensor
func (a Tensor) Close() {
	C.Tensor_Close(*a.T)
}

// MM multiplies each element of the input two tensors
func MM(a, b Tensor) Tensor {
	ct := C.MM(*a.T, *b.T)
	runtime.SetFinalizer(&ct, func(cf *C.Tensor) {
		C.Tensor_Close(*cf)
	})
	return Tensor{&ct}
}

// Sum returns the sum of all elements in the input tensor
func Sum(a Tensor) Tensor {
	ct := C.Sum(*a.T)
	runtime.SetFinalizer(&ct, func(cf *C.Tensor) {
		C.Tensor_Close(*cf)
	})
	return Tensor{&ct}
}
