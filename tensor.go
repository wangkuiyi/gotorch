//go:generate cgotorch/build.sh

package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	"fmt"
	"log"
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

// Save the tensor to a file
func (a Tensor) Save(path string) {
	C.Tensor_Save(C.Tensor(*a.T), C.CString(path))
}

// Load tensor from a file
func Load(path string) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Load(C.CString(path), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
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
	if len(shape) == 0 {
		return shape
	}
	MustNil(unsafe.Pointer(C.Tensor_Shape(C.Tensor(*a.T), (*C.int64_t)(unsafe.Pointer(&shape[0])))))
	return shape
}

// Dtype returns data type
func (a Tensor) Dtype() int8 {
	var t int8
	MustNil(unsafe.Pointer(C.Tensor_Dtype(C.Tensor(*a.T), (*C.int8_t)(unsafe.Pointer(&t)))))
	return t
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
func (a Tensor) To(device Device, dtype ...int8) Tensor {
	var t C.Tensor
	var d int8
	if len(dtype) == 0 {
		d = a.Dtype()
	} else {
		d = dtype[0]
	}
	MustNil(unsafe.Pointer(C.Tensor_To(C.Tensor(*a.T), device.T, C.int8_t(d), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// CUDA returns a Tensor on CUDA device
func (a Tensor) CUDA(device Device, nonBlocking bool) Tensor {
	var t C.Tensor
	n := int8(0)
	if nonBlocking {
		n = 1
	}
	MustNil(unsafe.Pointer(C.Tensor_CUDA(C.Tensor(*a.T), device.T, C.int8_t(n), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// CastTo cast tensor dtype
func (a Tensor) CastTo(dtype int8) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_CastTo(C.Tensor(*a.T), C.int8_t(dtype), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// CopyTo cast tensor dtype
func (a Tensor) CopyTo(device Device) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_CopyTo(C.Tensor(*a.T), device.T, &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// PinMemory returns a tensor in pinned memory. Pinned memory requires CUDA.
func (a Tensor) PinMemory() Tensor {
	if !IsCUDAAvailable() {
		return a
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_PinMemory(C.Tensor(*a.T), &t)))
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

// FromBlob returns a deep copy Tensor with the given data memory
func FromBlob(data unsafe.Pointer, dtype int8, sizes []int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_FromBlob(
		data,
		C.int8_t(dtype),
		(*C.int64_t)(unsafe.Pointer(&sizes[0])),
		C.int64_t(len(sizes)),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Index calls Tensor::index to return a single-element tensor of the element at
// the given index.
func (a Tensor) Index(index ...int64) Tensor {
	if int64(len(index)) != a.Dim() {
		log.Panicf("Index %v has length that differs from the tenosr dim %d", index, a.Dim())
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Index(
		C.Tensor(*a.T),
		(*C.int64_t)(unsafe.Pointer(&index[0])),
		C.int64_t(len(index)), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Reshape calls Tensor::reshape to return a tensor with the given shape
func (a Tensor) Reshape(sizes ...int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Reshape(
		C.Tensor(*a.T),
		(*C.int64_t)(unsafe.Pointer(&sizes[0])),
		C.int64_t(len(sizes)), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

func tensorListToSlice(ts []C.Tensor, cLength C.int64_t) []Tensor {
	length := int64(cLength)
	if length == 0 {
		return nil
	}
	results := make([]Tensor, length)
	for i := int64(0); i < length; i++ {
		results[i] = Tensor{
			T: (*unsafe.Pointer)(&ts[i]),
		}
		// ToDo: need to fix finalizer for tensor list
		//SetTensorFinalizer(results[i].T)
	}
	return results
}

// Split calls Tensor::split to return a slice of tensors
func (a Tensor) Split(splitSize int64, dim int64) []Tensor {
	shapes := a.Shape()
	resSize := (shapes[dim] + splitSize - 1) / splitSize
	fmt.Println("resSize: ", resSize)
	ts := make([]C.Tensor, resSize)
	var cLength C.int64_t
	MustNil(unsafe.Pointer(C.Tensor_Split(
		C.Tensor(*a.T),
		C.int64_t(splitSize),
		C.int64_t(dim),
		&ts[0],
		&cLength)))
	return tensorListToSlice(ts, cLength)
}

// Slice calls Tensor::slice to return a slice of tensors
func (a Tensor) Slice(dim int64, start int64, end int64, step int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Slice(
		C.Tensor(*a.T),
		C.int64_t(dim),
		C.int64_t(start),
		C.int64_t(end),
		C.int64_t(step),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Norm calls Tensor::norm to return a tensor norm
func (a Tensor) Norm(p int64, dim int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Norm(
		C.Tensor(*a.T),
		C.int64_t(p),
		C.int64_t(dim),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Unsqueeze calls Tensor::unsqueeze to return a tensor with a dimension of size one
func (a Tensor) Unsqueeze(dim int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Unsqueeze(
		C.Tensor(*a.T),
		C.int64_t(dim),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// LengthByShapes returns the length of a tensor by its shapes
func (a Tensor) LengthByShapes() (shapes []int64, length int64) {
	shapes = a.Shape()
	if len(shapes) == 0 {
		return shapes, 0
	}
	length = int64(1)
	for _, s := range shapes {
		length *= s
	}
	return shapes, length
}

func (a Tensor) Select(dim int64, index int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tensor_Select(
		C.Tensor(*a.T),
		C.int64_t(dim),
		C.int64_t(index),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}
