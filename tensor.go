//go:generate cgotorch/build.sh

package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

import (
	"fmt"
	"log"
	"unsafe"
)

// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T        *unsafe.Pointer
	Parents  []*Tensor
	RefCount int
}

//NewTensor creates a tensor
func NewTensor(t *unsafe.Pointer, parents ...*Tensor) *Tensor {
	res := &Tensor{
		T:       t,
		Parents: make([]*Tensor, 0),
	}
	for _, p := range parents {
		res.Parents = append(res.Parents, p)
		p.RefCount++
	}
	return res
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
func (a *Tensor) Detach() *Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Detach(C.Tensor(*a.T), &t)))
	return NewTensor((*unsafe.Pointer)(&t), a)
}

// String returns the Tensor as a string
func (a *Tensor) String() string {
	s := C.Tensor_String(C.Tensor(*a.T))
	r := C.GoString(s)
	C.FreeString(s)
	return r
}

// Print the tensor
func (a Tensor) Print() {
	C.Tensor_Print(C.Tensor(*a.T))
}

// Save the tensor to a file
func (a *Tensor) Save(path string) {
	C.Tensor_Save(C.Tensor(*a.T), C.CString(path))
}

// Close frees the C++ Tensor recursively
func (a *Tensor) Close() {
	if a.T == nil {
		return
	}
	a.RefCount--
	if a.RefCount <= 0 {
		C.Tensor_Close(C.Tensor(*a.T))
		fmt.Println("Call Tensor_Close")
		a.T = nil
	}
	for _, p := range a.Parents {
		p.Close()
	}
}

// Load tensor from a file
func Load(path string) *Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Load(C.CString(path), &t)))
	return NewTensor((*unsafe.Pointer)(&t))
}

// Dim returns dim
func (a *Tensor) Dim() int64 {
	var dim int64
	MustNil(unsafe.Pointer(C.Tensor_Dim(C.Tensor(*a.T), (*C.int64_t)(&dim))))
	return dim
}

// Shape returns shape
func (a *Tensor) Shape() []int64 {
	shape := make([]int64, a.Dim())
	if len(shape) == 0 {
		return shape
	}
	MustNil(unsafe.Pointer(C.Tensor_Shape(C.Tensor(*a.T), (*C.int64_t)(unsafe.Pointer(&shape[0])))))
	return shape
}

// Dtype returns data type
func (a *Tensor) Dtype() int8 {
	var t int8
	MustNil(unsafe.Pointer(C.Tensor_Dtype(C.Tensor(*a.T), (*C.int8_t)(unsafe.Pointer(&t)))))
	return t
}

// Backward compute the gradient of current tensor
func (a *Tensor) Backward() {
	C.Tensor_Backward(C.Tensor(*a.T))
}

// Grad returns a reference of the gradient
func (a *Tensor) Grad() *Tensor {
	t := C.Tensor_Grad(C.Tensor(*a.T))
	return NewTensor((*unsafe.Pointer)(&t), a)
}

// To returns a Tensor on the specified device with the same content as the a.
// If the specified device doesn't exist, To panics.
func (a *Tensor) To(device Device, dtype ...int8) *Tensor {
	var t C.Tensor
	var d int8
	if len(dtype) == 0 {
		d = a.Dtype()
	} else {
		d = dtype[0]
	}
	MustNil(unsafe.Pointer(C.Tensor_To(C.Tensor(*a.T), device.T, C.int8_t(d), &t)))
	return NewTensor((*unsafe.Pointer)(&t), a)
}

// CastTo cast tensor dtype
func (a *Tensor) CastTo(dtype int8) *Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_CastTo(C.Tensor(*a.T), C.int8_t(dtype), &t)))
	return NewTensor((*unsafe.Pointer)(&t), a)
}

// CopyTo cast tensor dtype
func (a *Tensor) CopyTo(device Device) *Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_CopyTo(C.Tensor(*a.T), device.T, &t)))
	return NewTensor((*unsafe.Pointer)(&t), a)
}

// PinMemory returns a tensor in pinned memory. Pinned memory requires CUDA.
func (a *Tensor) PinMemory() *Tensor {
	if !IsCUDAAvailable() {
		return a
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_PinMemory(C.Tensor(*a.T), &t)))
	return NewTensor((*unsafe.Pointer)(&t), a)
}

// SetData sets the tensor data held by b to a
func (a *Tensor) SetData(b *Tensor) {
	MustNil(unsafe.Pointer(C.Tensor_SetData(C.Tensor(*a.T), C.Tensor(*b.T))))
}

// To returns a Tensor on the specified device with the same content as the a.
// If the specified device doesn't exist, To panics.
func To(a *Tensor, device Device, dtype int8) *Tensor {
	return a.To(device, dtype)
}

// FromBlob returns a deep copy Tensor with the given data memory
func FromBlob(data unsafe.Pointer, dtype int8, sizes []int64) *Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_FromBlob(
		data,
		C.int8_t(dtype),
		(*C.int64_t)(unsafe.Pointer(&sizes[0])),
		C.int64_t(len(sizes)),
		&t)))
	return NewTensor((*unsafe.Pointer)(&t))
}

// Index calls Tensor::index to return a single-element tensor of the element at
// the given index.
func (a *Tensor) Index(index ...int64) *Tensor {
	if int64(len(index)) != a.Dim() {
		log.Panicf("Index %v has length that differs from the tenosr dim %d", index, a.Dim())
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Index(
		C.Tensor(*a.T),
		(*C.int64_t)(unsafe.Pointer(&index[0])),
		C.int64_t(len(index)), &t)))
	return NewTensor((*unsafe.Pointer)(&t), a)
}
