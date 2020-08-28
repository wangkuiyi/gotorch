package nn

import (
	"fmt"
	"log"
	"reflect"

	torch "github.com/wangkuiyi/gotorch"
)

// IModule is the interface of `Module`s
type IModule interface {
	// Train corresponds to torch.nn.Module.train(bool). It effects only
	// certain modules like Dropout and BatchNorm.
	Train(on bool)
	// IsTraining returns true if the module is in training mode
	IsTraining() bool
	// To corresponds to torch.nn.Module.to().  It recursively casts all
	// parameters to the given `dtype` and `device`.
	To(device torch.Device)
	// ZeroGrad corresponds to torch.nn.Module.zero_grad(). It recursively
	// zeros out the `grad` value of each registered parameter.
	ZeroGrad()
	// TODO(shendiaomo): Implement this.
	// String is for printing modules prettily.
	// String() string
}

// Module contains default implementation of `Module`s
type Module struct {
	outer IModule
	// Whether the module is in training mode.
	isTraining bool
	// The module's name (e.g. "LSTM").
	name string
}

// Init initializes a `Module`, using a `Module` that's not `Init`ed will panic
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
	must(m.outer != nil, "GoTorch requires defining modules via embedding a `Module` struct by value")
}

// Train enables "training" mode
func (m *Module) Train(on bool) {
	must(m.outer != nil, "GoTorch requires calling `Init` before using")
	m.isTraining = on
	sv := reflect.ValueOf(m.outer).Elem()
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)
		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for j := 0; j < v.Len(); j++ {
				must(v.CanInterface(),
					"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
				if m, ok := v.Index(j).Interface().(IModule); ok {
					if !reflect.ValueOf(m).IsNil() {
						m.Train(on)
					}
				}
			}
		} else {
			must(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
			if m, ok := v.Interface().(IModule); ok {
				if !reflect.ValueOf(m).IsNil() {
					m.Train(on)
				}
			}
		}
	}
}

// IsTraining returns true if the module is in training mode
func (m *Module) IsTraining() bool {
	return m.isTraining
}

// To recursively casts all parameters to the given `dtype` and `device`.
func (m *Module) To(device torch.Device) {
	must(m.outer != nil, "GoTorch requires calling `Init` before using")
	// TODO(shendiaomo): to be implemented after the `To` method of `Tensors` is ready
	moduleType := reflect.TypeOf((*IModule)(nil)).Elem()
	tensorType := reflect.TypeOf((*torch.Tensor)(nil)).Elem()
	sv := reflect.ValueOf(m.outer).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)
		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for j := 0; j < v.Len(); j++ {
				must(v.CanInterface(),
					"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
				if m, ok := v.Index(j).Interface().(IModule); ok {
					if !reflect.ValueOf(m).IsNil() {
						m.To(device)
					}
				}
			}
		} else if f.Type.Implements(moduleType) {
			if sv.Type() == moduleType && v.Addr() == reflect.ValueOf(m.outer).Addr() {
				// Skip `outer` itself
				continue
			}
			must(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
			if m, ok := v.Interface().(IModule); ok {
				if !reflect.ValueOf(m).IsNil() {
					m.To(device)
				}
			}
		} else if f.Type == tensorType {
			param := v.Interface().(torch.Tensor)
			if param.T != nil {
				param.SetData(param.To(device, param.Dtype()))
			}
		}
	}
}

// ZeroGrad recursively zeros out the `grad` value of each registered parameter.
func (m *Module) ZeroGrad() {
	must(m.outer != nil, "GoTorch modules requires calling `Init` before using")
	moduleType := reflect.TypeOf((*IModule)(nil)).Elem()
	tensorType := reflect.TypeOf((*torch.Tensor)(nil)).Elem()
	sv := reflect.ValueOf(m.outer).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)
		// TODO(shendiaomo): take reflect.Map into consideration
		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for j := 0; j < v.Len(); j++ {
				must(v.CanInterface(),
					"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
				if m, ok := v.Index(j).Interface().(IModule); ok {
					if !reflect.ValueOf(m).IsNil() {
						m.ZeroGrad()
					}
				}
			}
		} else if f.Type.Implements(moduleType) {
			if sv.Type() == moduleType && v.Addr() == reflect.ValueOf(m.outer).Addr() {
				// Skip `outer` itself
				continue
			}
			must(v.CanInterface(),
				"GoTorch requires exporting Module field %s.%s", sv.Type().Name(), f.Name)
			if m, ok := v.Interface().(IModule); ok {
				if !reflect.ValueOf(m).IsNil() {
					m.ZeroGrad()
				}
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

// TODO(shendiaomo): to be implemented
// String is for printing modules prettily
// func (m *Module) String() string {
// 	return m.name
// }

// NamedParameters returns trainable parameters (recursively) with their names
func (m *Module) NamedParameters() map[string]torch.Tensor {
	must(m.outer != nil, "GoTorch modules requires calling `Init` before using")
	r := make(map[string]torch.Tensor)
	visitTensors(m.outer, reflect.TypeOf(m.outer).Elem().Name(),
		makeTensorRecorder(r, true, false))
	return r
}

// NamedBuffers returns parameters (recursively) that are not trainable, with their names
func (m *Module) NamedBuffers() map[string]torch.Tensor {
	must(m.outer != nil, "GoTorch modules requires calling `Init` before using")
	r := make(map[string]torch.Tensor)
	visitTensors(m.outer, reflect.TypeOf(m.outer).Elem().Name(),
		makeTensorRecorder(r, false, true))
	return r
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

// Visitor is a function type supposed to be called by visitTensors.  The
// parameter f and v are a tensor-typed field in a given module. Returning
// non-nil eror breaks the recursive visiting process.
type Visitor func(f reflect.StructField, v reflect.Value, prefix string) error

func visitTensors(m IModule, prefix string, visitor Visitor) error {
	if reflect.ValueOf(m).IsNil() {
		return nil // No need to visit fields in a nil Module value.
	}

	moduleType := reflect.TypeOf((*IModule)(nil)).Elem()
	tensorType := reflect.TypeOf((*torch.Tensor)(nil)).Elem()

	sv := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)

		switch {
		case (v.Kind() == reflect.Slice || v.Kind() == reflect.Array):
			// The field is a slice or array.
			must(v.CanInterface(), "Please export slice and array field %s.%s",
				sv.Type().Name(), f.Name)
			for j := 0; j < v.Len(); j++ {
				visitTensors(v.Index(j).Interface().(IModule),
					fmt.Sprintf("%s.%s[%d]", prefix, f.Name, j), visitor)
			}

		case f.Type.Implements(moduleType):
			// The field is of type Module.
			must(v.CanInterface(), "Please export Module field %s.%s",
				sv.Type().Name(), f.Name)
			visitTensors(v.Interface().(IModule), prefix+"."+f.Name, visitor)

		case f.Type == tensorType:
			// The field is of type Tensor.
			must(v.CanInterface(), "Please export Tensor field %s.%s",
				v.Type().Name(), f.Name)
			if e := visitor(f, v, prefix); e != nil {
				return e
			}
		}
	}
	return nil
}

// makeTensorRecorder returns a visitor function that records a parameter and/or
// buffer tensor in a module into map record with key set to prefix+"."+f.Name.
func makeTensorRecorder(record map[string]torch.Tensor, param, buffer bool) Visitor {
	return func(f reflect.StructField, v reflect.Value, prefix string) error {
		tag := f.Tag.Get("gotorch")
		if (buffer && tag == "buffer") || (param && (tag == "param" || tag == "")) {
			// If the field is what we want.
			fv := v.Interface().(torch.Tensor)
			if fv.T != nil {
				// Don't record nil Tensor, for example,
				// unspecified bias of module Linear.
				record[prefix+"."+f.Name] = fv
			}
		}
		return nil
	}
}

// StateDict mimics torch.Module.state_dict(), which returns parameters and
// buffers with their (unique) names.
func (m *Module) StateDict() map[string]torch.Tensor {
	must(m.outer != nil, "GoTorch modules requires calling `Init` before using")
	r := make(map[string]torch.Tensor)
	visitTensors(m.outer, reflect.TypeOf(m.outer).Elem().Name(),
		makeTensorRecorder(r, true, true))
	return r
}

// SetStateDict sets the module's all tensor fields to values in sd.
func (m *Module) SetStateDict(sd map[string]torch.Tensor) error {
	must(m.outer != nil, "GoTorch modules requires calling `Init` before using")

	// SetStateDict requires that (1) entries in the map are all defined in
	// the module, and (2) parameters and buffers in the module are all in
	// the map.  To ensure (2), we go over module fields using Go
	// reflection, and for each field, we check the existence of the
	// corresponding entry in the map.  To ensure (2), we store fields in a
	// set, and check that every entry in the map is in the set.
	marks := make(map[string]int)

	e := visitTensors(m.outer, reflect.TypeOf(m.outer).Elem().Name(),
		makeTensorSetter(sd, marks))
	if e != nil {
		return e
	}

	for k := range sd {
		if _, ok := marks[k]; !ok {
			return fmt.Errorf("sd[%s] is not used to set any field", k)
		}
	}
	return nil
}

func makeTensorSetter(src map[string]torch.Tensor, marks map[string]int) Visitor {
	return func(f reflect.StructField, v reflect.Value, prefix string) error {
		k := prefix + "." + f.Name
		t, ok := src[k]
		if !ok {
			return fmt.Errorf("Cannot find field %s in the value map", k)
		}
		v.Set(reflect.ValueOf(t))
		marks[k]++
		return nil
	}
}

func must(condition bool, fmtStr string, args ...interface{}) {
	if !condition {
		// Use logPanicf to be recoverable and enable stack trace
		log.Panicf(fmtStr, args...)
	}
}
