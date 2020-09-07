package functional

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestFunctionalBatchNorm(t *testing.T) {
	input := torch.RandN([]int64{10, 20}, true)
	w := torch.RandN([]int64{20}, true)
	r := BatchNorm(input, torch.Tensor{}, torch.Tensor{}, w, torch.Tensor{}, true, 0.1, 0.1)
	assert.NotNil(t, r.T)
}

/*
filters = torch.randn(8,4,3,3)
inputs = torch.randn(1,4,5,5)
o = F.conv2d(inputs, filters, padding=1)
*/
func TestFunctionalConv2d(t *testing.T) {
	filters := torch.RandN([]int64{8, 4, 3, 3}, false)
	inputs := torch.RandN([]int64{1, 4, 5, 5}, false)
	var bias torch.Tensor
	o := Conv2d(inputs, filters, bias, []int64{1, 1}, []int64{1, 1}, []int64{1, 1}, 1)
	assert.NotNil(t, o.T)
	assert.Equal(t, []int64{1, 8, 5, 5}, o.Shape())
}

/*
import torch.nn.functional as F
inputs = torch.randn(1, 4, 5, 5)
weights = torch.randn(4, 8, 3, 3)
o = F.conv_transpose2d(inputs, weights, padding=1)
*/
func TestFunctionalConvTranspose2d(t *testing.T) {
	input := torch.RandN([]int64{1, 4, 5, 5}, false)
	weight := torch.RandN([]int64{4, 8, 3, 3}, false)
	var bias torch.Tensor
	stride := []int64{1, 1}
	padding := []int64{1, 1}
	outputPadding := []int64{0, 0}
	groups := int64(1)
	dilation := []int64{1, 1}
	out := ConvTranspose2d(input, weight, bias,
		stride, padding, outputPadding, groups, dilation)
	assert.NotNil(t, out.T)
	assert.Equal(t, []int64{1, 8, 5, 5}, out.Shape())
}

/*
# input is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
output = F.nll_loss(F.log_softmax(input), target)
output.backward()
input.grad.shape
*/
func TestFunctionalNllLoss(t *testing.T) {
	input := torch.RandN([]int64{3, 5}, true)
	target := torch.NewTensor([]int64{1, 0, 4})
	var weight torch.Tensor
	output := NllLoss(LogSoftmax(input, -1), target, weight, -100, "mean")
	output.Backward()
	assert.Equal(t, []int64{3, 5}, input.Grad().Shape())
}

// >>> torch.nn.functional.log_softmax(torch.tensor([[-0.5, -1.], [1., 0.5]]), dim=1)
// tensor([[-0.4741, -0.9741],
//         [-0.4741, -0.9741]])
func TestFunctionalLogSoftmax(t *testing.T) {
	r := LogSoftmax(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		1)
	g := "-0.4741 -0.9741\n-0.4741 -0.9741\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

func TestFunctionalMaxPool2d(t *testing.T) {
	input := torch.RandN([]int64{20, 16, 50, 32}, false)
	out := MaxPool2d(input, []int64{3, 2}, []int64{2, 1}, []int64{0, 0}, []int64{1, 1}, false)
	assert.NotNil(t, out.T)
}

func TestFunctionalAdaptiveAvgPool2d(t *testing.T) {
	input := torch.RandN([]int64{1, 64, 8, 9}, false)
	out := AdaptiveAvgPool2d(input, []int64{5, 7})
	assert.NotNil(t, out.T)
}

// >>> input = torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
// >>> target = torch.tensor([[6.,5.],[4.,3.],[2.,1.]])
// >>> loss = F.binary_cross_entropy(torch.sigmoid(input), target)
// >>> print(loss)
// tensor(-5.7473)
func TestFunctionalBinaryCrossEntropy(t *testing.T) {
	input := torch.NewTensor([][]float64{{1, 2}, {3, 4}, {5, 6}})
	target := torch.NewTensor([][]float64{{6, 5}, {4, 3}, {2, 1}})
	var weight torch.Tensor
	loss := BinaryCrossEntropy(torch.Sigmoid(input), target, weight, "mean")
	assert.Equal(t, "-5.74731\n[ CPUDoubleType{} ]", loss.String())
}

// >>> import torch
// >>> import torch.nn.functional as F
// >>> F.relu(torch.tensor([-1, -0.5, 0, 0.5, 1.]))
// tensor([0.0000, 0.0000, 0.0000, 0.5000, 1.0000])
func TestFunctionalRelu(t *testing.T) {
	e := " 0.0000\n 0.0000\n 0.0000\n 0.5000\n 1.0000\n[ CPUDoubleType{5} ]"
	o := Relu(torch.NewTensor([]float64{-1, -0.5, 0, 0.5, 1}), false)
	assert.Equal(t, e, o.String())

	p := Relu(torch.NewTensor([]float64{-1, -0.5, 0, 0.5, 1}), true)
	assert.Equal(t, e, p.String())
}

// >>> import torch
// >>> import torch.nn.functional as F
// >>> F.leaky_relu(torch.tensor([-1, -0.5, 0, 0.5, 1.]))
// tensor([-0.0100, -0.0050,  0.0000,  0.5000,  1.0000])
func TestFunctionalLeakyRelu(t *testing.T) {
	e := "-0.0100\n-0.0050\n 0.0000\n 0.5000\n 1.0000\n[ CPUDoubleType{5} ]"
	o := LeakyRelu(torch.NewTensor([]float64{-1, -0.5, 0, 0.5, 1}), 0.01, false)
	assert.Equal(t, e, o.String())

	p := LeakyRelu(torch.NewTensor([]float64{-1, -0.5, 0, 0.5, 1}), 0.01, true)
	assert.Equal(t, e, p.String())
}

// >>> import torch
// i>>> import torch.nn.functional as F
// >>> F.linear(torch.tensor([[1.,2.],[3.,4.]]), torch.tensor([[1.,2.],[3.,4.]]), torch.tensor([1.,2.]))
// tensor([[ 6., 13.],
//         [12., 27.]])
func TestFunctionalLinear(t *testing.T) {
	o := Linear(
		torch.NewTensor([][]float64{{1, 2}, {3, 4}}),
		torch.NewTensor([][]float64{{1, 2}, {3, 4}}),
		torch.NewTensor([]float64{1, 2}))
	assert.Equal(t, "  6  13\n 12  27\n[ CPUDoubleType{2,2} ]", o.String())
}

// >>> target = torch.tensor([0,3,4], dtype=torch.int64)
// >>> input = torch.tensor([[0.,1.,2.,1.,0.],[0.,1.,2.,1.,0.],[0.,1.,2.,1.,0.]])
// >>> F.cross_entropy(input, target)
// tensor(2.3630)
func TestFunctionalCrossEntropy(t *testing.T) {
	target := torch.NewTensor([]int64{0, 3, 4})
	input := torch.NewTensor([][]float64{
		{0, 1, 2, 1, 0.},
		{0, 1, 2, 1, 0.},
		{0, 1, 2, 1, 0.}})
	var weight torch.Tensor
	o := CrossEntropy(input, target, weight, -100, "mean")
	assert.Equal(t, "2.36302\n[ CPUDoubleType{} ]", o.String())
}
