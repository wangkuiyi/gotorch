package functional

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestTranspose2d(t *testing.T) {
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
}

func TestBatchNorm(t *testing.T) {
	input := torch.RandN([]int64{10, 20}, true)
	w := torch.RandN([]int64{20}, true)
	r := BatchNorm(input, torch.Tensor{}, torch.Tensor{}, w, torch.Tensor{}, true, 0.1, 0.1)
	assert.NotNil(t, r.T)
}

func TestBinaryCrossEntropy(t *testing.T) {
	input := torch.RandN([]int64{3, 2}, true)
	target := torch.Rand([]int64{3, 2}, false)
	loss := BinaryCrossEntropy(torch.Sigmoid(input), target, torch.Tensor{}, "mean")
	assert.NotNil(t, loss.T)
	loss.Backward()
}

func TestMaxPool2d(t *testing.T) {
	input := torch.RandN([]int64{20, 16, 50, 32})
	out := MaxPool2d([]int64{3, 2}, []int64{2, 1}, []int64{1, 1}, false)
	assert.NotNil(t, out.T)
}

func TestAdaptiveAvgPool2d(t *testing.T) {
	input := torch.RandN([]int64{1, 64, 8, 9})
	out := AdaptiveAvgPool2d(input, []int64{5, 7})
	assert.NotNil(t, out.T)
}
