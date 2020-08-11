package functional

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestTranspose2d(t *testing.T) {
	input := torch.RandN([]int{1, 4, 5, 5}, false)
	weight := torch.RandN([]int{4, 8, 3, 3}, false)
	var bias torch.Tensor
	stride := []int{1, 1}
	padding := []int{1, 1}
	outputPadding := []int{0, 0}
	groups := 1
	dilation := []int{1, 1}
	out := ConvTranspose2d(input, weight, bias,
		stride, padding, outputPadding, groups, dilation)
	assert.NotNil(t, out.T)
}

func TestBatchNorm(t *testing.T) {
	input := torch.RandN([]int{10, 20}, true)
	w := torch.RandN([]int{20}, true)
	r := BatchNorm(input, w, torch.Tensor{}, torch.Tensor{}, torch.Tensor{}, true, 0.1, 0.1)
	assert.NotNil(t, r.T)
}
