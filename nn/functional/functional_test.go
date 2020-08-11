package functional

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestTranspose2d(t *testing.T) {
	input := torch.RandN([]int64{1, 1, 1}, false)
	weight := torch.RandN([]int64{1, 3, 3}, false)
	var bias torch.Tensor
	stride := []int64{1}
	padding := []int64{0}
	outputPadding := []int64{0}
	groups := int64(1)
	dilation := []int64{1}
	out := ConvTranspose2d(input, weight, bias,
		stride, padding, outputPadding, groups, dilation)
	assert.NotNil(t, out.T)
}

func TestBatchNorm(t *testing.T) {
	input := torch.RandN([]int64{10, 20}, true)
	w := torch.RandN([]int64{20}, true)
	r := BatchNorm(input, w, torch.Tensor{}, torch.Tensor{}, torch.Tensor{}, true, 0.1, 0.1, true)
	assert.NotNil(t, r.T)
}
