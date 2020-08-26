package transforms

import (
	"fmt"
	"reflect"
)

// Transform interface
type Transform interface{}

// ComposeTransformer composes transforms together
type ComposeTransformer struct {
	// Transform function should implement a `Do` method, which accepts any argument type
	// and returns a value.
	Transforms []interface{}
}

// Compose returns a ComposeTransformer
func Compose(transforms ...interface{}) *ComposeTransformer {
	return &ComposeTransformer{Transforms: transforms}
}

// Run executes the transforms sequentially
func (t *ComposeTransformer) Run(inputs ...interface{}) interface{} {
	if len(t.Transforms) == 0 {
		panic("Cannot call Run() on an empty ComposeTransformer")
	}
	run := reflect.ValueOf(t.Transforms[0]).MethodByName("Run")
	if !run.IsValid() {
		panic(fmt.Sprintf("GoTorch required exporting `Run` receiver on %s", reflect.TypeOf(t.Transforms[0])))
	}
	input := getInterfaceInputs(run.Call(getReflectInputs(inputs)))
	for i := 1; i < len(t.Transforms); i++ {
		run := reflect.ValueOf(t.Transforms[i]).MethodByName("Run")
		if !run.IsValid() {
			panic(fmt.Sprintf("GoTorch required exporting `Run` receiver on %s", reflect.TypeOf(t.Transforms[0])))
		}
		input = getInterfaceInputs(run.Call(getReflectInputs(input)))
	}
	if len(input) != 1 {
		panic("The last transfrom in Compose must have exactly one return value")
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
