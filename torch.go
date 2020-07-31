package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"reflect"
	"runtime"
	"unsafe"
)

// Optimizer struct
type Optimizer struct {
	Opt *C.Optimizer
}

// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	ct := C.RandN(C.int(rows), C.int(cols), C.int(rg))
	runtime.SetFinalizer(&ct, func(cf *C.Tensor) {
		C.Tensor_Close(*cf)
	})
	return Tensor{&ct}
}

// NewSGDOpt creates a SGD Optimizer
func NewSGDOpt(lr, momentum, dampening, weightDecay float64, nesterov bool) Optimizer {
	nt := 0
	if nesterov {
		nt = 1
	}
	co := C.SGD(C.double(lr), C.double(momentum), C.double(dampening),
		C.double(weightDecay), C.int(nt))
	runtime.SetFinalizer(&co, func(cf *C.Optimizer) {
		C.Optimizer_Close(*cf)
	})
	return Optimizer{&co}
}

// AddParameters adds parameters
func (opt Optimizer) AddParameters(tensors []Tensor) {
	CT := []unsafe.Pointer{}
	for _, t := range tensors {
		CT = append(CT, unsafe.Pointer(*t.T))
	}
	p := (*reflect.SliceHeader)(unsafe.Pointer(&CT)).Data
	C.AddParameters(*opt.Opt, (*C.Tensor)(unsafe.Pointer(p)), C.int(len(CT)))
}

// ZeroGrad reset gradients to zero
func (opt Optimizer) ZeroGrad() {
	C.ZeroGrad(*opt.Opt)
}

// Step updates parameters
func (opt Optimizer) Step() {
	C.Step(*opt.Opt)
}

// Close the optimizer
func (opt Optimizer) Close() {
	C.Optimizer_Close(*opt.Opt)
}

// Model struct
type Model struct {
	Parameters []Tensor
}

// NewModel creates a model instance
func NewModel() *Model {
	return &Model{
		Parameters: make([]Tensor, 0),
	}
}

// Module interface
type Module interface {
	Forward(x Tensor) Tensor
}

// linear struct
type linear struct {
	InFeatures  int
	OutFeatures int
	Weight      Tensor
	Bias        Tensor
}

// Linear creates a linear instance
func Linear(model *Model, in int, out int, bias bool) Module {
	l := &linear{
		InFeatures:  in,
		OutFeatures: out,
	}
	l.Weight = RandN(in, out, true)
	model.Parameters = append(model.Parameters, l.Weight)
	if bias {
		l.Bias = RandN(out, 1, true)
		model.Parameters = append(model.Parameters, l.Bias)
	}
	return l
}

// Forward executes the calculation
func (l *linear) Forward(x Tensor) Tensor {
	return MM(x, l.Weight)
}
