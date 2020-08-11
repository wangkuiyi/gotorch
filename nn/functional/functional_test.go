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
