package module

import (
	"log"
	"reflect"
)

// Module interface
type Module interface {
	Forward(x Tensor) Tensor
}

// GetNamedParameters returns parameters in the module recursively.
func GetNamedParameters(m Module) map[string]Tensor {
	r := make(map[string]Tensor)
	getNamedNonNilTensors(m, reflect.TypeOf(m).Elem().Name(), true, false, r)
	return r
}

func getNamedNonNilTensors(m Module, prefix string, param, buffer bool, r map[string]Tensor) {
	moduleType := reflect.TypeOf((*Module)(nil)).Elem()

	sv := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)

		if f.Type.Implements(moduleType) {
			if !v.CanInterface() {
				log.Fatalf("GoTorch requires exporting Module field %s.%s",
					sv.Type().Name(), f.Name)
			}
			getNamedNonNilTensors(v.Interface().(Module),
				prefix+"."+f.Name, param, buffer, r)
		} else {
			recordNonNilTensor(f, v, prefix, r, param, buffer)
		}
	}
}

// If field f is a parameter or buffer field and the value v is not a nil
// tensor, insert v into map r with key is prefix+"."+f.Name.
func recordNonNilTensor(f reflect.StructField, v reflect.Value,
	prefix string, r map[string]Tensor, param, buffer bool) {

	tensorType := reflect.TypeOf((*Tensor)(nil)).Elem()
	if f.Type != tensorType {
		return // Either parameter or buffer is of type Tensor.
	}

	tag := f.Tag.Get("gotorch")
	if !buffer && tag == "buffer" {
		return // Don't wants a buffer but this field is one.
	}
	if !param && (tag == "param" || tag == "") {
		return // Don't wants a parameter but this field is one.
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
	r := make(map[string]Tensor)
	getNamedNonNilTensors(m, reflect.TypeOf(m).Elem().Name(), true, true, r)
	for _, t := range r {
		t.Close()
	}
}
