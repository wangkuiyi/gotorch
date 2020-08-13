package nn

import (
	"reflect"

	torch "github.com/wangkuiyi/gotorch"
)

// SequentialModule is a list of `Module`s that acts as a `Module` itself.
type SequentialModule struct {
	Module
	Modules []IModule
}

// Sequential returns a new `SequentialModule` instance.
func Sequential(modules ...IModule) *SequentialModule {
	r := &SequentialModule{Module: Module{isTraining: true}, Modules: modules}
	r.Init(r)
	return r
}

// Forward feeds `inputs` to the first module and then chains outputs to inputs, returning the last output.
func (s *SequentialModule) Forward(inputs ...interface{}) interface{} {
	if len(s.Modules) == 0 {
		panic("Cannot call forward() on an empty Sequential")
	}
	forward := reflect.ValueOf(s.Modules[0]).MethodByName("Forward")
	input := getInterfaceInputs(forward.Call(getReflectInputs(inputs)))
	for i := 1; i < len(s.Modules); i++ {
		forward := reflect.ValueOf(s.Modules[i]).MethodByName("Forward")
		input = getInterfaceInputs(forward.Call(getReflectInputs(input)))
	}
	if len(input) != 1 {
		panic("The last module in Sequential must have exactly one return value")
	}
	return input[0]
}

func getReflectInputs(inputs []interface{}) (r []reflect.Value) {
	for _, i := range inputs {
		r = append(r, reflect.ValueOf(i))
	}
	return r
}

func getInterfaceInputs(prevReturned []reflect.Value) (r []interface{}) {
	for _, v := range prevReturned {
		r = append(r, v.Interface())
	}
	return r
}

// FunctionalModule wraps a function in a `Module`.
type FunctionalModule struct {
	Module
	function func(torch.Tensor) torch.Tensor
}

// Functional returns a new `Functional` instance.
func Functional(f func(torch.Tensor) torch.Tensor) *FunctionalModule {
	r := &FunctionalModule{Module: Module{isTraining: true}, function: f}
	r.Init(r)
	return r
}

// Forward feeds the `input` tensor to the underlying function.
func (f *FunctionalModule) Forward(input torch.Tensor) torch.Tensor {
	return f.function(input)
}
