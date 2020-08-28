package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestBatchNorm2d(t *testing.T) {
	b := BatchNorm2d(100, 1e-5, 0.1, true, true)
	x := torch.RandN([]int64{20, 100, 35, 45}, false)
	output := b.Forward(x)
	assert.NotNil(t, output.T)
}
