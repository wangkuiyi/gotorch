package functional

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTranspose2d(t *testing.T) {
	input := torch.RandN([]int{1, 1, 1}, false)
	weight := torch.RandN([]int{1, 3, 3}, false)
	var bias torch.Tensor
	stride := []int{1}
	padding := []int{0}
	outputPadding := []int{0}
	groups := 1
	dilation := []int{1}
	out := torch.ConvTranspose2d(input, weight, bias,
		stride, padding, outputPadding, groups, dilation)
	a := assert.New(t)
	a.NotNil(out.T)
}
