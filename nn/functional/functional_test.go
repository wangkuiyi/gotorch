package functional

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestBatchNorm(t *testing.T) {
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
func TestConv2d(t *testing.T) {
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
func TestConvTranspose2d(t *testing.T) {
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
*/
func TestNllLoss(t *testing.T) {
}

func TestBinaryCrossEntropy(t *testing.T) {
	input := torch.RandN([]int64{3, 2}, true)
	target := torch.Rand([]int64{3, 2}, false)
	loss := BinaryCrossEntropy(torch.Sigmoid(input), target, torch.Tensor{}, "mean")
	assert.NotNil(t, loss.T)
	loss.Backward()
}

func TestMaxPool2d(t *testing.T) {
	input := torch.RandN([]int64{20, 16, 50, 32}, false)
	out := MaxPool2d(input, []int64{3, 2}, []int64{2, 1}, []int64{0, 0}, []int64{1, 1}, false)
	assert.NotNil(t, out.T)
}

func TestAdaptiveAvgPool2d(t *testing.T) {
	input := torch.RandN([]int64{1, 64, 8, 9}, false)
	out := AdaptiveAvgPool2d(input, []int64{5, 7})
	assert.NotNil(t, out.T)
}
