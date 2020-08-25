package transforms

import (
	"fmt"
	"reflect"
)

// Transform interface
type Transform interface{}

// NormalizeTransformer corresponds to torchvision.transforms.html#Normalize. It
// implements Go interface gotorch/data.Transform.
type NormalizeTransformer struct {
	Mean, Stddev float64
}

// Normalize returns normalize transformer
func Normalize(mean float64, stddev float64) *NormalizeTransformer {
	return &NormalizeTransformer{mean, stddev}
}

// ComposeTransforms composes transforms together
type ComposeTransforms struct {
	// Transform function should implement a `Do` method, which accepts any argument type
	// and returns a value.
	Transforms []interface{}
}

// Compose returns a ComposeTransforms
func Compose(transforms ...interface{}) *ComposeTransforms {
	return &ComposeTransforms{Transforms: transforms}
}

// Do executes the transforms sequentially
func (t *ComposeTransforms) Do(inputs ...interface{}) interface{} {
	if len(t.Transforms) == 0 {
		panic("Cannot call Do() on an empty ComposeTransforms")
	}
	do := reflect.ValueOf(t.Transforms[0]).MethodByName("Do")
	if !do.IsValid() {
		panic(fmt.Sprintf("GoTorch required exporting `Do` receiver on %s", reflect.TypeOf(t.Transforms[0])))
	}
	input := getInterfaceInputs(do.Call(getReflectInputs(inputs)))
	for i := 1; i < len(t.Transforms); i++ {
		do := reflect.ValueOf(t.Transforms[i]).MethodByName("Do")
		if !do.IsValid() {
			panic(fmt.Sprintf("GoTorch required exporting `Do` receiver on %s", reflect.TypeOf(t.Transforms[0])))
		}
		input = getInterfaceInputs(do.Call(getReflectInputs(input)))
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
