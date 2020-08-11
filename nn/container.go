package nn

import (
	torch "github.com/wangkuiyi/gotorch"
	"reflect"
)

// Sequential is a list of `Module`s that acts as a `Module` itself.
type Sequential struct {
	Module
	Modules []IModule
}

// NewSequential returns a new `Sequential` instance.
func NewSequential(modules ...IModule) *Sequential {
	return &Sequential{Module: Module{isTraining: true}, Modules: modules}
}

// Forward feeds `inputs` to the first module and then chains outputs to inputs, returning the last output.
func (s *Sequential) Forward(inputs ...interface{}) interface{} {
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

// Functional wraps a function in a `Module`.
type Functional struct {
	Module
	function func(torch.Tensor) torch.Tensor
}

// NewFunctional returns a new `Functional` instance.
func NewFunctional(f func(torch.Tensor) torch.Tensor) *Functional {
	return &Functional{Module: Module{isTraining: true}, function: f}
}

// Forward feeds the `input` tensor to the underlying function.
func (f *Functional) Forward(input torch.Tensor) torch.Tensor {
	return f.function(input)
}
