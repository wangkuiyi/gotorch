package jit

// #cgo CFLAGS: -I ${SRCDIR}/..
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
// #include <stdlib.h>
import "C"

import (
	torch "github.com/wangkuiyi/gotorch"
	"unsafe"
)

type Module struct {
	M *unsafe.Pointer
}

type IValue struct {
	I *unsafe.Pointer
}

// LoadJITModule loads a model from a *.pt file
func LoadJITModule(modelPath string) *Module {
	var c C.Module
	torch.MustNil(unsafe.Pointer(
		C.loadModule(C.CString(modelPath), &c),
	))
	return &Module{M: (*unsafe.Pointer)(&c)}
}

// Forward runs the forward pass of the model
func (m *Module) Forward(input torch.Tensor) *IValue {
	var c C.IValue
	torch.MustNil(unsafe.Pointer(
		C.forwardModule((C.Module)(*m.M), (C.Tensor)(*input.T), &c),
	))
	return &IValue{I: (*unsafe.Pointer)(&c)}
}

// IsTuple returns true if the IValue is a tuple
func (i *IValue) IsTuple() bool {
	return bool(C.IValue_isTuple((C.IValue)(*i.I)))
}

// IsTensor returns true if the IValue is a tensor
func (i *IValue) IsTensor() bool {
	return bool(C.IValue_isTensor((C.IValue)(*i.I)))
}

// ToTuple converts the IValue to a tuple (go slice)
func (i *IValue) ToTuple() []*IValue {
	var c *C.IValue
	var cLength C.int
	torch.MustNil(unsafe.Pointer(
		C.IValue_toTuple((C.IValue)(*i.I), &c, &cLength),
	))
	length := int(cLength)
	results := make([]*IValue, length)
	for i := 0; i < length; i++ {
		results[i] = &IValue{
			I: (*unsafe.Pointer)(unsafe.Pointer(uintptr(unsafe.Pointer(c)) + uintptr(i)*unsafe.Sizeof(c))),
		}
	}
	return results
}

// ToTensor converts the IValue to a tensor
func (i *IValue) ToTensor() torch.Tensor {
	var c C.Tensor
	torch.MustNil(unsafe.Pointer(
		C.IValue_toTensor((C.IValue)(*i.I), &c),
	))
	return torch.Tensor{T: (*unsafe.Pointer)(&c)}
}
