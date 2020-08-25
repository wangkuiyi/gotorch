package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

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

func TestAddI(t *testing.T) {
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

// >>> torch.sigmoid(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.3775, 0.2689],
//         [0.7311, 0.6225]])
func TestSigmoid(t *testing.T) {
	r := torch.Sigmoid(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	g := ` 0.3775  0.2689
 0.7311  0.6225
[ CPUFloatType{2,2} ]`
	assert.Equal(t, g, r.String())
}

// >>> torch.mean(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor(0.)
func TestMean(t *testing.T) {
	r := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}).Mean()
	// BUG: The result should be 0.
	g := `0
[ CPUFloatType{} ]`
	assert.Equal(t, g, r.String())
}

// >>> torch.nn.functional.log_softmax(torch.tensor([[-0.5, -1.], [1., 0.5]]), dim=1)
// tensor([[-0.4741, -0.9741],
//         [-0.4741, -0.9741]])
func TestLogSoftmax(t *testing.T) {
	r := torch.LogSoftmax(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		1)
	g := `-0.4741 -0.9741
-0.4741 -0.9741
[ CPUFloatType{2,2} ]`
	assert.Equal(t, g, r.String())
}

// >>> t = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> s = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> t+s
// tensor([[-1., -2.],
//         [ 2.,  1.]])
func TestAdd(t *testing.T) {
	r := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	s := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	q := r.Add(s, 1)
	g := `-1 -2
 2  1
[ CPUFloatType{2,2} ]`
	assert.Equal(t, g, q.String())
}

// >>> torch.transpose(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([[-0.5000,  1.0000],
//         [-1.0000,  0.5000]])
func TestTranspose(t *testing.T) {
	r := torch.Transpose(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0, 1)
	g := `-0.5000  1.0000
-1.0000  0.5000
[ CPUFloatType{2,2} ]`
	assert.Equal(t, g, r.String())
}

// >>> torch.flatten(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([-0.5000, -1.0000,  1.0000,  0.5000])
func TestFlatten(t *testing.T) {
	r := torch.Flatten(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0, 1)
	g := `-0.5000
-1.0000
 1.0000
 0.5000
[ CPUFloatType{4} ]`
	assert.Equal(t, g, r.String())
}
