package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"reflect"
	"unsafe"
)

// Optimizer struct
type Optimizer struct {
	Opt C.Optimizer
}

// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	return NewTensor(unsafe.Pointer(C.RandN(C.int(rows), C.int(cols), C.int(rg))))
}

// NewSGDOpt creates a SGD Optimizer
func NewSGDOpt(lr, momentum, dampening, weightDecay float64, nesterov bool) Optimizer {
	nt := 0
	if nesterov {
		nt = 1
	}
	return Optimizer{
		C.SGD(C.double(lr), C.double(momentum), C.double(dampening),
			C.double(weightDecay), C.int(nt))}
}

// AddParameters adds parameters
func (opt Optimizer) AddParameters(tensors []Tensor) {
	CT := []unsafe.Pointer{}
	for _, t := range tensors {
		CT = append(CT, unsafe.Pointer(t.T))
	}
	p := (*reflect.SliceHeader)(unsafe.Pointer(&CT)).Data
	C.AddParameters(opt.Opt, (*C.Tensor)(unsafe.Pointer(p)), C.int(len(CT)))
}

// ZeroGrad reset gradients to zero
func (opt Optimizer) ZeroGrad() {
	C.ZeroGrad(opt.Opt)
}

// Step updates parameters
func (opt Optimizer) Step() {
	C.Step(opt.Opt)
}

// Close the optimizer
func (opt Optimizer) Close() {
	C.Optimizer_Close(opt.Opt)
}

// Model struct
type Model struct {
	Parameters []Tensor
	Variables  []Tensor
}

// NewModel creates a model instance
func NewModel() *Model {
	return &Model{
		Parameters: make([]Tensor, 0),
		Variables:  make([]Tensor, 0),
	}
}

// CloseVariables release memory of variables
func (m *Model) CloseVariables() {
	for _, v := range m.Variables {
		v.Close()
	}
	m.Variables = make([]Tensor, 0)
}

// CloseParameters release memory of parameters
func (m *Model) CloseParameters() {
	for _, p := range m.Parameters {
		p.Close()
	}
}

// Linear struct
type Linear struct {
	M           *Model
	InFeatures  int
	OutFeatures int
	Weight      Tensor
	Bias        Tensor
}

// NewLinear creates a linear layer
func NewLinear(m *Model, in int, out int, bias bool) *Linear {
	l := &Linear{
		M:           m,
		InFeatures:  in,
		OutFeatures: out,
	}
	l.Weight = RandN(in, out, true)
	l.M.Parameters = append(l.M.Parameters, l.Weight)
	if bias {
		l.Bias = RandN(out, 1, true)
		l.M.Parameters = append(l.M.Parameters, l.Bias)
	}
	return l
}

// Forward executes the calculation
func (l *Linear) Forward(x Tensor) Tensor {
	t := MM(x, l.Weight)
	l.M.Variables = append(l.M.Variables, t)
	return t
}
