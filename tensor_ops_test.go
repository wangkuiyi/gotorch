package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

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

func TestTensorMean(t *testing.T) {
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

func TestStack(t *testing.T) {
	t1 := torch.RandN([]int64{2, 3}, false)
	t2 := torch.RandN([]int64{2, 3}, false)
	out := torch.Stack([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{2, 2, 3}, out.Shape())
}

// >>> torch.relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.0000, 0.0000],
//         [1.0000, 0.5000]])
func TestRelu(t *testing.T) {
	r := torch.Relu(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	g := ` 0.0000  0.0000
 1.0000  0.5000
[ CPUFloatType{2,2} ]`
	assert.Equal(t, g, r.String())
}

// >>> torch.nn.functional.leaky_relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[-0.0050, -0.0100],
//         [ 1.0000,  0.5000]])
func TestLeakyRelu(t *testing.T) {
	r := torch.LeakyRelu(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0.01)
	g := `-0.0050 -0.0100
 1.0000  0.5000
[ CPUFloatType{2,2} ]`
	assert.Equal(t, g, r.String())
}
