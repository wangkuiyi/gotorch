package jit

// #cgo CFLAGS: -I ${SRCDIR}/..
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	torch "github.com/wangkuiyi/gotorch"
	"unsafe"
)

type Module struct {
	M *unsafe.Pointer
}

// LoadJITModule loads a model from a *.pt file
func LoadJITModule(modelPath string) *Module {
	var c C.Module
	torch.MustNil(unsafe.Pointer(
		C.loadModule(C.CString(modelPath), &c),
	))
	SetModuleFinalizer((*unsafe.Pointer)(&c))
	return &Module{M: (*unsafe.Pointer)(&c)}
}

// Forward runs the forward pass of the model
func (m *Module) Forward(input torch.Tensor) *IValue {
	var c C.IValue
	torch.MustNil(unsafe.Pointer(
		C.forwardModule((C.Module)(*m.M), (C.Tensor)(*input.T), &c),
	))
	SetIValueFinalizer((*unsafe.Pointer)(&c))
	return &IValue{I: (*unsafe.Pointer)(&c)}
}
