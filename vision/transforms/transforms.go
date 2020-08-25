package transforms

import (
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

// SequentialTransform indicates executing the transform functions sequentially
type SequentialTransform struct {
	Transforms []interface{}
}

// Sequential returns a SequentialTransform
func Sequential(transforms ...interface{}) *SequentialTransform {
	t := &SequentialTransform{Transforms: transforms}
	return t
}

// Do executes the transforms sequentially
func (t *SequentialTransform) Do(inputs ...interface{}) interface{} {
	if len(t.Transforms) == 0 {
		panic("Cannot call Do() on an empty SequentialTransforms")
	}
	do := reflect.ValueOf(t.Transforms[0]).MethodByName("Do")
	input := getInterfaceInputs(do.Call(getReflectInputs(inputs)))
	for i := 1; i < len(t.Transforms); i++ {
		do := reflect.ValueOf(t.Transforms[i]).MethodByName("Do")
		input = getInterfaceInputs(do.Call(getReflectInputs(input)))
	}
	if len(input) != 1 {
		panic("The last transfrom in Sequential must have exactly one return value")
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
