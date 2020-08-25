package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestConv2d(t *testing.T) {
	c := Conv2d(16, 33, 3, 2, 0, 1, 1, true, "zeros")
	x := torch.RandN([]int64{20, 16, 50, 100}, false)
	output := c.Forward(x)
	assert.NotNil(t, output)
}

func TestConvTranspose2d(t *testing.T) {
	c := ConvTranspose2d(16, 33, 3, 2, 0, 1, 1, true, 1, "zeros")
	x := torch.RandN([]int64{20, 16, 50, 100}, false)
	output := c.Forward(x)
	assert.NotNil(t, output.T)
}
