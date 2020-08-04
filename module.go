package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"log"
	"reflect"
)

// Module interface
type Module interface {
	Forward(x Tensor) Tensor
}

type linear struct {
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
	if bias {
		l.Bias = RandN(out, 1, true)
	}
	return l
}

// Forward method
func (l *linear) Forward(x Tensor) Tensor {
	return MM(x, l.Weight)
}

func GetNamedParameters(m Module) map[string]Tensor {
	return getNamedParamsOrBuffers(m, true)
}

func GetNamedBuffers(m Module) map[string]Tensor {
	return getNamedParamsOrBuffers(m, false)
}

func getNamedParamsOrBuffers(m Module, param bool) map[string]Tensor {
	r := make(map[string]Tensor)
	visitModuleFields(m, reflect.TypeOf(m).Elem().Name(),
		func(f reflect.StructField, v reflect.Value, p string) {
			recordParamOrBuffer(f, v, p, r, param)
		})
	return r
}

// recordParamOrBuffer is the only implementation of this function type.
type moduleFieldVisitor func(f reflect.StructField, v reflect.Value, p string)

var (
	tensorType = reflect.TypeOf((*Tensor)(nil)).Elem()
	moduleType = reflect.TypeOf((*Module)(nil)).Elem()
)

func visitModuleFields(m Module, prefix string, fn moduleFieldVisitor) {
	sv := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)

		if f.Type.Implements(moduleType) {
			if !v.CanInterface() {
				log.Fatalf("GoTorch requires exporting Module field %s.%s",
					sv.Type().Name(), f.Name)
			}
			visitModuleFields(v.Interface().(Module),
				prefix+"."+f.Name, fn)
		} else {
			fn(f, v, prefix)
		}
	}
}

// If field f is a parameter or buffer field and the value v is not a nil
// tensor, insert v into map r with key is prefix+"."+f.Name.
func recordParamOrBuffer(f reflect.StructField, v reflect.Value,
	prefix string, r map[string]Tensor, param bool) {

	if f.Type != tensorType {
		return // Either parameter or buffer is of type Tensor.
	}

	tag := f.Tag.Get("gotorch")
	if param && tag == "buffer" {
		return // Wants a parameter but this field is a buffer.
	}
	if !param && (tag == "param" || tag == "") {
		return // Wants a buffer but this field is a parameter.
	}

	if !v.CanInterface() {
		log.Fatalf("GoTorch requires exporting Tensor field %s.%s",
			v.Type().Name(), f.Name)
	}

	fv := v.Interface().(Tensor)
	if fv.T == nil {
		return // Don't record nil Tensor
	}

	r[prefix+"."+f.Name] = fv
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
