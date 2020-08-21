package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

func ExampleTensor() {
	t := torch.RandN([]int64{10, 100}, false)
	t.Close()
	t.Close()
	// Output:
}

func TestLogSoftmax(t *testing.T) {
	a := assert.New(t)
	x := torch.RandN([]int64{1, 6}, false)
	out := x.LogSoftmax(1)
	// TODO(yancey1989): convert torchTensor as Go slice, that we can
	// check the value.
	a.NotNil(out.T)
}

func TestSqueeze(t *testing.T) {
	x := torch.RandN([]int64{2, 1, 2, 1, 2}, false)
	y := torch.Squeeze(x)
	assert.NotNil(t, y.T)
	z := torch.Squeeze(x, 1)
	assert.NotNil(t, z.T)
}

func TestItem(t *testing.T) {
	x := torch.RandN([]int64{1}, false)
	r := x.Item()
	assert.NotNil(t, r)
}

func TestDetach(t *testing.T) {
	x := torch.RandN([]int64{1}, true)
	y := x.Detach()
	assert.NotNil(t, y.T)
	initializer.Zeros(&y)
	assert.Equal(t, float32(0.0), x.Item())
}

func TestMean(t *testing.T) {
	x := torch.RandN([]int64{2, 3}, true)
	y := x.Mean()
	z := y.Item()
	assert.NotNil(t, z)
}

func TestAdd(t *testing.T) {
	x := torch.RandN([]int64{2, 3}, false)
	y := torch.RandN([]int64{2, 3}, false)
	z := torch.Add(x, y, 1)
	x.AddI(y, 1)
	assert.True(t, torch.Equal(x, z))
}
