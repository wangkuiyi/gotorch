package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestRandN(t *testing.T) {
	a := torch.RandN([]int64{10, 100}, false)
	assert.Equal(t, []int64{10, 100}, a.Shape())
}

func TestRand(t *testing.T) {
	a := torch.Rand([]int64{50, 100}, false)
	assert.Equal(t, []int64{50, 100}, a.Shape())
}

func TestEmpty(t *testing.T) {
	a := torch.Empty([]int64{50, 10}, false)
	assert.Equal(t, []int64{50, 10}, a.Shape())
}

func TestOnes(t *testing.T) {
	a := torch.Ones([]int64{3, 3}, false)
	assert.Equal(t, ` 1  1  1
 1  1  1
 1  1  1
[ CPUFloatType{3,3} ]`, a.String())

	a = torch.Ones([]int64{1}, false)
	assert.Equal(t, ` 1
[ CPUFloatType{1} ]`, a.String())

	a = torch.Ones([]int64{1}, true)
	assert.Equal(t, ` 1
[ CPUFloatType{1} ]`, a.String())
}

func TestEye(t *testing.T) {
	a := torch.Eye(3, 3, false)
	assert.Equal(t, ` 1  0  0
 0  1  0
 0  0  1
[ CPUFloatType{3,3} ]`, a.String())

	a = torch.Eye(3, 2, false)
	assert.Equal(t, ` 1  0
 0  1
 0  0
[ CPUFloatType{3,2} ]`, a.String())

	a = torch.Eye(3, 1, false)
	assert.Equal(t, ` 1
 0
 0
[ CPUFloatType{3,1} ]`, a.String())

	a = torch.Eye(3, 1, true)
	assert.Equal(t, ` 1
 0
 0
[ CPUFloatType{3,1} ]`, a.String())
}

func TestFull(t *testing.T) {
	a := torch.Full([]int64{3, 3}, 1, false)
	assert.Equal(t, ` 1  1  1
 1  1  1
 1  1  1
[ CPUFloatType{3,3} ]`, a.String())

	assert.True(t, torch.Equal(a, torch.Ones([]int64{3, 3}, false)))

	a = torch.Full([]int64{3, 3}, 0, false)
	assert.Equal(t, ` 0  0  0
 0  0  0
 0  0  0
[ CPUFloatType{3,3} ]`, a.String())

	a = torch.Full([]int64{1}, 100, false)
	assert.Equal(t, ` 100
[ CPUFloatType{1} ]`, a.String())

	a = torch.Full([]int64{1}, 100, true)
	assert.Equal(t, ` 100
[ CPUFloatType{1} ]`, a.String())
}

func TestArange(t *testing.T) {
	a := torch.Arange(0, 5, 1, false)
	assert.Equal(t, ` 0
 1
 2
 3
 4
[ CPUFloatType{5} ]`, a.String())

	a = torch.Arange(0, 5, 2, false)
	assert.Equal(t, ` 0
 2
 4
[ CPUFloatType{3} ]`, a.String())

	a = torch.Arange(0, 5, 3, false)
	assert.Equal(t, ` 0
 3
[ CPUFloatType{2} ]`, a.String())

	a = torch.Arange(0, 5, 5, false)
	assert.Equal(t, ` 0
[ CPUFloatType{1} ]`, a.String())

	a = torch.Arange(0, 5, 5, true)
	assert.Equal(t, ` 0
[ CPUFloatType{1} ]`, a.String())
}

func TestLinspace(t *testing.T) {
	a := torch.Linspace(0, 5, 6, false)
	assert.Equal(t, ` 0
 1
 2
 3
 4
 5
[ CPUFloatType{6} ]`, a.String())

	a = torch.Linspace(0, 5, 3, false)
	assert.Equal(t, ` 0.0000
 2.5000
 5.0000
[ CPUFloatType{3} ]`, a.String())

	a = torch.Linspace(0, 5, 3, true)
	assert.Equal(t, ` 0.0000
 2.5000
 5.0000
[ CPUFloatType{3} ]`, a.String())

	a = torch.Linspace(0, 5, 1, false)
	assert.Equal(t, ` 0
[ CPUFloatType{1} ]`, a.String())
}

func TestLogspace(t *testing.T) {
	a := torch.Logspace(0, 5, 6, 10, false)
	assert.Equal(t, `      1
     10
    100
   1000
  10000
 100000
[ CPUFloatType{6} ]`, a.String())
	assert.Equal(t, torch.Float, a.Dtype())

	a = torch.Logspace(0, 5, 3, 10, false)
	assert.Equal(t, ` 1.0000e+00
 3.1623e+02
 1.0000e+05
[ CPUFloatType{3} ]`, a.String())

	a = torch.Logspace(0, 5, 1, 10, false)
	assert.Equal(t, ` 1
[ CPUFloatType{1} ]`, a.String())

	a = torch.Logspace(0, 5, 6, 2, false)
	assert.Equal(t, `  1
  2
  4
  8
 16
 32
[ CPUFloatType{6} ]`, a.String())

	a = torch.Logspace(0, 5, 6, 2, true)
	assert.Equal(t, `  1
  2
  4
  8
 16
 32
[ CPUFloatType{6} ]`, a.String())
}
