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
	outer IModule
	// Whether the module is in training mode.
	isTraining bool
	// The module's name (e.g. "LSTM").
	name string
}

// Init initializes a `Module`, using a `Module` that's not `Init`ed is undefined behavior
// Example:
//
// type MyModel struct {
// 	Module
// }
//
// func NewMyModule() *MyModel {
// 	r := &MyModel{}
// 	r.Init(r)
// 	return r
// }
func (m *Module) Init(outer IModule) {
	if m.outer != nil {
		return
	}
	moduleType := reflect.TypeOf(m).Elem()
	fv := reflect.ValueOf(outer).Elem()
	for i := 0; i < fv.NumField(); i++ {
		v := fv.Field(i)
		f := fv.Type().Field(i)
		if f.Type == moduleType && f.Name == moduleType.Name() {
			if v.Addr() == reflect.ValueOf(m) {
				// Calling Init in a valid Module: struct{*Module} or struct{Module}
				m.outer = outer
				m.isTraining = true
			}
		}
	}
	torchCheck(m.outer != nil, "GoTorch requires defining modules via embedding a `Module` struct by value")
}

// Train enables "training" mode
func (m *Module) Train(on bool) {
	m.isTraining = on
	sv := reflect.ValueOf(m.outer).Elem()
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)
		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for j := 0; j < v.Len(); j++ {
				torchCheck(v.CanInterface(),
					"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
				if m, ok := v.Index(j).Interface().(IModule); ok {
					m.Train(on)
				}
			}
		} else {
			torchCheck(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
			if m, ok := v.Interface().(IModule); ok {
				m.Train(on)
			}
		}
	}
}

// IsTraining returns true if the module is in training mode
func (m *Module) IsTraining() bool {
	return m.isTraining
}

// To recursively casts all parameters to the given `dtype` and `device`.
func (m *Module) To(device Device, dtype Dtype, nonBlocking bool) {
	// TODO(shendiaomo): to be implemented after the `To` method of `Tensors` is ready
}

// ZeroGrad recursively zeros out the `grad` value of each registered parameter.
func (m *Module) ZeroGrad() {
	moduleType := reflect.TypeOf((*IModule)(nil)).Elem()
	tensorType := reflect.TypeOf((*torch.Tensor)(nil)).Elem()
	sv := reflect.ValueOf(m.outer).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)
		// TODO(shendiaomo): take reflect.Map into consideration
		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for j := 0; j < v.Len(); j++ {
				torchCheck(v.CanInterface(),
					"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
				if m, ok := v.Index(j).Interface().(IModule); ok {
					m.ZeroGrad()
				}
			}
		} else if f.Type.Implements(moduleType) {
			if sv.Type() == moduleType && v.Addr() == reflect.ValueOf(m.outer).Addr() {
				// Skip `outer` itself
				continue
			}
			torchCheck(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
			if m, ok := v.Interface().(IModule); ok {
				m.ZeroGrad()
			}
		} else if f.Type == tensorType {
			/* TODO(shendiaomo): implement `Grad`, `Defined` and `Detach`
			grad := v.Interface().(torch.Tensor).Grad()
			if grad.Defined() {
				grad = grad.Detach()
				grad.Zero_()
			}
			*/
		}
	}
}

// String is for printing modules prettily
func (m *Module) String() string {
	// TODO(shendiaomo): to be implemented
	return m.name
}

// NamedParameters returns trainable parameters (recursively) with their names
func (m *Module) NamedParameters() map[string]torch.Tensor {
	r := make(map[string]torch.Tensor)
	getNamedNonNilTensors(m.outer, reflect.TypeOf(m.outer).Elem().Name(), true, false, r)
	return r
}

// NamedBuffers returns parameters (recursively) that are not trainable, with their names
func (m *Module) NamedBuffers() map[string]torch.Tensor {
	r := make(map[string]torch.Tensor)
	getNamedNonNilTensors(m.outer, reflect.TypeOf(m.outer).Elem().Name(), false, true, r)
	return r
}

func getNamedNonNilTensors(m IModule, prefix string, param, buffer bool, r map[string]torch.Tensor) {
	moduleType := reflect.TypeOf((*IModule)(nil)).Elem()
	sv := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)
		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			torchCheck(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
			for j := 0; j < v.Len(); j++ {
				getNamedNonNilTensors(v.Index(j).Interface().(IModule),
					fmt.Sprintf("%s.%s[%d]", prefix, f.Name, j), param, buffer, r)
			}
		} else if f.Type.Implements(moduleType) {
			torchCheck(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
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

	torchCheck(v.CanInterface(),
		"GoTorch requires exporting Tensor field %s.%s", v.Type().Name(), f.Name)
	fv := v.Interface().(torch.Tensor)
	if fv.T == nil {
		return // Don't record nil Tensor
	}

	r[prefix+"."+f.Name] = fv
}

// Parameters returns trainable parameters (recursively)
func (m *Module) Parameters() []torch.Tensor {
	result := make([]torch.Tensor, 0)
	n := m.NamedParameters()
	for _, v := range n {
		result = append(result, v)
	}
	return result
}

// Buffers returns parameters (recursively) that are not trainable
func (m *Module) Buffers() []torch.Tensor {
	result := make([]torch.Tensor, 0)
	n := m.NamedBuffers()
	for _, v := range n {
		result = append(result, v)
	}
	return result
}

func torchCheck(condition bool, fmtStr string, args ...interface{}) {
	if !condition {
		// Use logPanicf to be recoverable and enable stack trace
		log.Panicf(fmtStr, args...)
	}
}
