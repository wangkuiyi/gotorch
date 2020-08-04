package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"reflect"
)

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

func GetNamedParameters(m Module) map[string]Tensor {
	r := make(map[string]Tensor)

	moduleType := reflect.TypeOf((*Module)(nil)).Elem()
	tensorType := reflect.TypeOf((*Tensor)(nil)).Elem()

	v := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < v.NumField(); i++ {
		fn := v.Type().Field(i).Name
		ft := v.Type().Field(i).Type
		fg := v.Type().Field(i).Tag
		fv := v.Field(i).Interface()

		if ft.Implements(moduleType) {
			rr := GetNamedParameters(fv.(Module))
			for k, v := range rr {
				r[fn+"."+k] = v
			}
		} else if ft == tensorType && fg.Get("gotorch") != "buffer" {
			if ct := fv.(Tensor).T; ct != nil {
				r[fn] = fv.(Tensor)
			}
		}
	}
	return r
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

// CloseModule closes the module
func CloseModule(m Module) {
	params := GetParameters(m)
	for _, p := range params {
		p.Close()
	}
}
