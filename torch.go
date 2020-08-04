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
	t := C.RandN(C.int(rows), C.int(cols), C.int(rg))
	setTensorFinalizer(&t)
	return Tensor{&t}
}

// NewSGDOpt creates a SGD Optimizer
func NewSGDOpt(lr, momentum, dampening, weightDecay float64, nesterov bool) Optimizer {
	nt := 0
	if nesterov {
		nt = 1
	}
	sgd := C.SGD(C.double(lr), C.double(momentum), C.double(dampening),
		C.double(weightDecay), C.int(nt))
	runtime.SetFinalizer(&sgd, func(p *C.Optimizer) { C.Optimizer_Close(*p) })
	return Optimizer{&sgd}
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

// Module interface
type Module interface {
	Forward(x Tensor) Tensor
}

// Model struct
type Model struct {
	parameters map[string]Tensor
	buffers    map[string]Tensor
	modules    map[string]Module
}

// RegisterBuffer registers a buffer to the model
func (m *Model) RegisterBuffer(s string, t Tensor) {
	if m.buffers == nil {
		m.buffers = make(map[string]Tensor)
	}
	m.buffers[s] = t
}

// RegisterParameter registers a parameter to the model
func (m *Model) RegisterParameter(s string, p Tensor) {
	if m.parameters == nil {
		m.parameters = make(map[string]Tensor)
	}
	m.parameters[s] = p
}

// RegisterModule registers a module to the model
func (m *Model) RegisterModule(s string, module Module) {
	if m.modules == nil {
		m.modules = make(map[string]Module)
	}
	m.modules[s] = module
}

type linear struct {
	Model
	InFeatures  int
	OutFeatures int
	Weight      Tensor
	Bias        Tensor
}

// Linear creates a linear instance
func Linear(in int, out int, bias bool) Module {
	l := &linear{
		InFeatures:  in,
		OutFeatures: out,
	}
	l.Weight = RandN(in, out, true)
	l.RegisterParameter("Weight", l.Weight)
	if bias {
		l.Bias = RandN(out, 1, true)
		l.RegisterParameter("Bias", l.Bias)
	}
	return l
}

// Forward method
func (l *linear) Forward(x Tensor) Tensor {
	return MM(x, l.Weight)
}

// GetChildrenModules returns children modules recursively
func GetChildrenModules(m Module) map[string]Module {
	result := make(map[string]Module)
	model := reflect.ValueOf(m).Elem().Field(0).Interface().(Model)
	for name, cm := range model.modules {
		result[name] = cm
		res := GetChildrenModules(cm)
		for k, v := range res {
			result[k] = v
		}
	}
	return result
}

// GetNamedParameters returns named parameters
func GetNamedParameters(m Module) map[string]Tensor {
	result := make(map[string]Tensor)
	model := reflect.ValueOf(m).Elem().Field(0).Interface().(Model)
	for k, v := range model.parameters {
		result[k] = v
	}
	for moduleName, v := range GetChildrenModules(m) {
		cmodel := reflect.ValueOf(v).Elem().Field(0).Interface().(Model)
		for paramName, cv := range cmodel.parameters {
			name := moduleName + "_" + paramName
			result[name] = cv
		}
	}
	return result
}

// GetParameters returns parameters
func GetParameters(m Module) []Tensor {
	result := make([]Tensor, 0)
	n := GetNamedParameters(m)
	for _, v := range n {
		result = append(result, v)
	}
	return result
}
