package nn

import (
	"fmt"
	"log"
	"reflect"

	torch "github.com/wangkuiyi/gotorch"
)

// Dtype is the data type of scalars
type Dtype int // TODO(shendiaomo): this is a placeholder to be defined later

// DeviceType is the type of devices
type DeviceType int // TODO(shendiaomo): this is a placeholder to be defined latert

// DeviceIndex is the index of available devices
type DeviceIndex int16

// Device represents a a compute device on which a tensor is located.
type Device struct {
	typ DeviceType
	idx DeviceIndex
}

// IModule is the interface of `Module`s
type IModule interface {
	// Train enables "training" mode
	Train(on bool)
	// IsTraining returns true if the module is in training mode
	IsTraining() bool
	// To recursively casts all parameters to the given `dtype` and `device`.
	To(device Device, dtype Dtype, nonBlocking bool)
	// ZeroGrad recursively zeros out the `grad` value of each registered parameter.
	ZeroGrad()
	// String is for printing modules prettily
	String() string
}

// Module contains default implementation of `Module`s
type Module struct {
	// Whether the module is in training mode.
	isTraining bool
	// The module's name (e.g. "LSTM").
	name string
}

// Train enables "training" mode
func (m *Module) Train(on bool) {
	m.isTraining = on
}

// IsTraining returns true if the module is in training mode
func (m *Module) IsTraining() bool {
	return m.isTraining
}

// To recursively casts all parameters to the given `dtype` and `device`.
func (m *Module) To(device Device, dtype Dtype, nonBlocking bool) {
	// TODO(shendiaomo): to be implemented
}

// ZeroGrad recursively zeros out the `grad` value of each registered parameter.
func (m *Module) ZeroGrad() {
	// TODO(shendiaomo): to be implemented
}

// String is for printing modules prettily
func (m *Module) String() string {
	// TODO(shendiaomo): to be implemented
	return m.name
}

// GetNamedParameters returns parameters in the module recursively.
func GetNamedParameters(m IModule) map[string]torch.Tensor {
	r := make(map[string]torch.Tensor)
	getNamedNonNilTensors(m, reflect.TypeOf(m).Elem().Name(), true, false, r)
	return r
}

func getNamedNonNilTensors(m IModule, prefix string, param, buffer bool, r map[string]torch.Tensor) {
	moduleType := reflect.TypeOf((*IModule)(nil)).Elem()
	sv := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)

		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for j := 0; j < v.Len(); j++ {
				if !v.Index(j).CanInterface() {
					log.Fatalf("GoTorch requires exporting Module field %s.%s",
						sv.Type().Name(), f.Name)
				}
				getNamedNonNilTensors(v.Index(j).Interface().(IModule),
					fmt.Sprintf("%s.%s[%d]", prefix, f.Name, j), param, buffer, r)
			}
		} else if f.Type.Implements(moduleType) {
			if !v.CanInterface() {
				log.Fatalf("GoTorch requires exporting Module field %s.%s",
					sv.Type().Name(), f.Name)
			}
			getNamedNonNilTensors(v.Interface().(IModule),
				prefix+"."+f.Name, param, buffer, r)
		} else {
			recordNonNilTensor(f, v, prefix, r, param, buffer)
		}
	}
}

// If field f is a parameter or buffer field and the value v is not a nil
// tensor, insert v into map r with key is prefix+"."+f.Name.
func recordNonNilTensor(f reflect.StructField, v reflect.Value,
	prefix string, r map[string]torch.Tensor, param, buffer bool) {

	tensorType := reflect.TypeOf((*torch.Tensor)(nil)).Elem()
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

	fv := v.Interface().(torch.Tensor)
	if fv.T == nil {
		return // Don't record nil Tensor
	}

	r[prefix+"."+f.Name] = fv
}

// GetParameters returns trainable parameters to an optimizer
func GetParameters(m IModule) []torch.Tensor {
	result := make([]torch.Tensor, 0)
	n := GetNamedParameters(m)
	for _, v := range n {
		result = append(result, v)
	}
	return result
}

// CloseModule closes the module
func CloseModule(m IModule) {
	r := make(map[string]torch.Tensor)
	getNamedNonNilTensors(m, reflect.TypeOf(m).Elem().Name(), true, true, r)
	for _, t := range r {
		t.Close()
	}
}
