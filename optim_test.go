package gotorch

import (
	"math"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

// CompareFloat compares two float32 number
func CompareFloat(a float64, b float64, tolerance float64) bool {
	diff := math.Abs(a - b)
	mean := math.Abs(a+b) / 2.0
	if math.IsNaN(diff / mean) {
		return true
	}
	return (diff / mean) < tolerance
}

// CompareFloatArray compares two float32/64 array
func CompareFloatArray(a interface{}, b interface{}, tolerance float64) bool {
	vala := reflect.ValueOf(a)
	valb := reflect.ValueOf(b)
	for i := 0; i < vala.Len(); i++ {
		if !CompareFloat(vala.Index(i).Float(), valb.Index(i).Float(), tolerance) {
			return false
		}
	}
	return true
}

func TestOptim(t *testing.T) {
	lr := 0.1
	b := RandN(4, 1, true)
	param1 := b.ToSlice()
	opt := SGD(lr, 0, 0, 0, false)
	opt.AddParameters([]Tensor{b})

	a := RandN(3, 4, false)
	c := MM(a, b)
	d := Sum(c)

	opt.ZeroGrad()
	d.Backward()
	grad := b.Grad().ToSlice()
	opt.Step()
	param2 := b.ToSlice()

	expectedParam := make([]float32, len(param1))
	for i := 0; i < len(expectedParam); i++ {
		expectedParam[i] = param1[i] - float32(lr)*grad[i]
	}
	assert.True(t, CompareFloatArray(expectedParam, param2, 0.0001))
}
