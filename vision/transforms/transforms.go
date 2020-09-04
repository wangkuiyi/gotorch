package transforms

import (
	"fmt"
	"reflect"
)

// Transform interface
// Note: this interface only used in wrapping C dataset
type Transform interface{}

// ComposeTransformer composes transforms together
type ComposeTransformer struct {
	// Transform function should implement a `Run` method that does the real computation.
	Transforms []interface{}
}

// Compose returns a ComposeTransformer
func Compose(transforms ...interface{}) *ComposeTransformer {
	return &ComposeTransformer{Transforms: transforms}
}

// Run executes the transformers sequentially
func (t *ComposeTransformer) Run(inputs ...interface{}) interface{} {
	for _, transform := range t.Transforms {
		run := reflect.ValueOf(transform).MethodByName("Run")
		if !run.IsValid() {
			panic(fmt.Sprintf("GoTorch required exporting `Run` receiver on %s", reflect.TypeOf(transform)))
		}
		inputs = getInterfaceInputs(run.Call(getReflectInputs(inputs)))
	}
	if len(inputs) != 1 {
		panic("The last transform in Compose must have exactly one return value")
	}
	return inputs[0]
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
