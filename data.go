package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdlib.h>
// #include "cgotorch.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// Dataset interface
type Dataset struct {
	T C.CDataset
}

// Transform interface
type Transform interface{}

// Normalize transform struct
type Normalize struct {
	T unsafe.Pointer
}

// Stack transform struct
type Stack struct {
	T unsafe.Pointer
}

// NewMnist returns MNIST dataset
func NewMnist(dataRoot string) *Dataset {
	cstr := C.CString(dataRoot)
	defer C.free(unsafe.Pointer(cstr))
	return &Dataset{C.CMnist(cstr)}
}

// NewNormalize returns normalize transformer
func NewNormalize(mean float64, stddev float64) *Normalize {
	return &Normalize{unsafe.Pointer(C.CNormalize(C.double(mean), C.double(stddev)))}
}

// NewStack returns Stack tranformer
func NewStack() *Stack {
	return &Stack{unsafe.Pointer(C.CStack())}
}

// AddTransforms adds a slice of Transform
func (d *Dataset) AddTransforms(transforms []Transform) {
	for _, trans := range transforms {
		switch v := trans.(type) {
		case *Normalize:
			C.AddNormalize(d.T, (C.CTransform)(trans.(*Normalize).T))
		case *Stack:
			C.AddStack(d.T, (C.CTransform)(trans.(*Stack).T))
		default:
			panic(fmt.Sprintf("unsupposed transform type: %T", v))
		}
	}
}
