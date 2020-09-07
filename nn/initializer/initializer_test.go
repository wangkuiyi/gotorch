package initializer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestManualSeed(t *testing.T) {
	ManualSeed(1)
	x := torch.RandN([]int64{1}, false)
	assert.Equal(t, float32(0.66135216), x.Item().(float32))
}

func TestNormal(t *testing.T) {
	var x torch.Tensor
	assert.Panics(t, func() { Normal(&x, 0.1, 0.2) })

	x = torch.Empty([]int64{2, 3}, false)
	Normal(&x, 0.1, 0.2)
	assert.NotNil(t, x.T)
}

func TestUniform(t *testing.T) {
	var x torch.Tensor
	assert.Panics(t, func() { Uniform(&x, 11.1, 22.2) })

	x = torch.Empty([]int64{2, 3}, false)
	Uniform(&x, 11, 22)
	forAllElements(t, x, func(elem interface{}) bool {
		return float32(11.1) <= elem.(float32)
	})
	forAllElements(t, x, func(elem interface{}) bool {
		return elem.(float32) < float32(22.2)
	})
}

func TestZeros(t *testing.T) {
	var x torch.Tensor
	assert.Panics(t, func() { Zeros(&x) })

	x = torch.Empty([]int64{2, 3}, false)
	Zeros(&x)
	forAllElements(t, x, func(elem interface{}) bool {
		return float32(0) == elem.(float32)
	})
}

func TestOnes(t *testing.T) {
	var x torch.Tensor
	assert.Panics(t, func() { Ones(&x) })

	x = torch.Empty([]int64{2, 3}, false)
	Ones(&x)
	forAllElements(t, x, func(elem interface{}) bool {
		return float32(1) == elem.(float32)
	})
}

func forAllElements(t *testing.T, x torch.Tensor, ck func(elem interface{}) bool) {
	shape := x.Shape()
	idx := make([]int64, len(shape))

	increment := func(idx, shape []int64) bool {
		i := 0
		for {
			idx[i]++
			if idx[i] >= shape[i] {
				if i+1 >= len(shape) {
					return false // no increment any more
				}
				idx[i] = 0
				i++
			} else {
				break
			}
		}
		return true // successfully increased idx
	}

	for {
		if !ck(x.Index(idx...).Item()) {
			t.Fatalf("forAllElements failed at index %v", idx)
		}
		if !increment(idx, shape) {
			break
		}
	}
}
