package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

import (
	"unsafe"
)

// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T *unsafe.Pointer
}

// MustNil asserts error to be nil
func MustNil(err unsafe.Pointer) {
	if err != nil {
		msg := C.GoString((*C.char)(err))
		C.FreeString((*C.char)(err))
		panic(msg)
	}
}

// Detach tensor.detach
func (a *Tensor) Detach() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Detach(C.Tensor(*a.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// String returns the Tensor as a string
func (a Tensor) String() string {
	s := C.Tensor_String(C.Tensor(*a.T))
	r := C.GoString(s)
	C.FreeString(s)
	return r
}

// Print the tensor
func (a Tensor) Print() {
	C.Tensor_Print(C.Tensor(*a.T))
}

// Close the tensor
func (a *Tensor) Close() {
	if a.T != nil {
		C.Tensor_Close(C.Tensor(*a.T))
		a.T = nil
	}
}

// Save the tensor to a file
func (a Tensor) Save(path string) {
	C.Tensor_Save(C.Tensor(*a.T), C.CString(path))
}

// Dim returns dim
func (a Tensor) Dim() int64 {
	var dim int64
	MustNil(unsafe.Pointer(C.Tensor_Dim(C.Tensor(*a.T), (*C.int64_t)(&dim))))
	return dim
}

// Shape returns shape
func (a Tensor) Shape() []int64 {
	shape := make([]int64, a.Dim())
	MustNil(unsafe.Pointer(C.Tensor_Shape(C.Tensor(*a.T), (*C.int64_t)(unsafe.Pointer(&shape[0])))))
	return shape
}

// Dtype returns data type
func (a Tensor) Dtype() int8 {
	var t int8
	MustNil(unsafe.Pointer(C.Tensor_Dtype(C.Tensor(*a.T), (*C.int8_t)(unsafe.Pointer(&t)))))
	return t
}

// Item torch.item
func (a Tensor) Item() float32 {
	var t float32
	MustNil(unsafe.Pointer(C.Item(C.Tensor(*a.T), (*C.float)(&t))))
	return t
}

// View returns a new Tensor with the same data but of a different shape
func (a Tensor) View(shape []int64) Tensor {
	return View(a, shape)
}

// Backward compute the gradient of current tensor
func (a Tensor) Backward() {
	C.Tensor_Backward(C.Tensor(*a.T))
}

// Grad returns a reference of the gradient
func (a Tensor) Grad() Tensor {
	t := C.Tensor_Grad(C.Tensor(*a.T))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// To returns a Tensor on the specified device with the same content as the a.
// If the specified device doesn't exist, To panics.
func (a Tensor) To(device Device, dtype int8) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_To(C.Tensor(*a.T), device.T, C.int8_t(dtype), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// SetData sets the tensor data held by b to a
func (a Tensor) SetData(b Tensor) {
	MustNil(unsafe.Pointer(C.Tensor_SetData(C.Tensor(*a.T), C.Tensor(*b.T))))
}

// To returns a Tensor on the specified device with the same content as the a.
// If the specified device doesn't exist, To panics.
func To(a Tensor, device Device, dtype int8) Tensor {
	return a.To(device, dtype)
}

// Equal compares two tensors by their content.
func Equal(a, b Tensor) bool {
	var r int64
	MustNil(unsafe.Pointer(C.Equal(C.Tensor(*a.T), C.Tensor(*b.T), (*C.int64_t)(&r))))
	return r != 0
}

// FromBlob creating a Tensor with the give data memory
func FromBlob(data unsafe.Pointer, dtype int8, sizes []int64) Tensor {
	var t C.Tensor
	C.Tensor_FromBlob(data, C.int8_t(dtype), (*C.int64_t)(unsafe.Pointer(&sizes[0])), C.int64_t(len(sizes)), &t)
	return Tensor{(*unsafe.Pointer)(&t)}
}
