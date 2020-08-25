package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

// >>> c = torch.nn.Conv2d(16, 33, 3, 2, 0, 1, 1, True, "zeros")
// >>> x = torch.randn(20, 16, 50, 100, requires_grad=False)
// >>> y=c(x)
// >>> y.shape
// torch.Size([20, 33, 24, 49])
func TestConv2d(t *testing.T) {
	c := Conv2d(16, 33, 3, 2, 0, 1, 1, true, "zeros")
	x := torch.RandN([]int64{20, 16, 50, 100}, false)
	output := c.Forward(x)
	assert.NotNil(t, output.T)
	assert.Equal(t, []int64{20, 33, 24, 49}, output.Shape())
}

// >>> c = torch.nn.ConvTranspose2d(16, 33, 3, 2, 0, 1, 1, True, 1, 'zeros')
// >>> x = torch.randn(20, 16, 50, 100)
// >>> y = c(x)
// >>> y.shape
// torch.Size([20, 33, 102, 202])
func TestConvTranspose2d(t *testing.T) {
	c := ConvTranspose2d(16, 33, 3, 2, 0, 1, 1, true, 1, "zeros")
	x := torch.RandN([]int64{20, 16, 50, 100}, false)
	output := c.Forward(x)
	assert.NotNil(t, output.T)
	assert.Equal(t, []int64{20, 33, 102, 202}, output.Shape())
}
