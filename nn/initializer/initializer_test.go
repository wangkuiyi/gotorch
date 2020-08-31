package initializer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestManualSeed(t *testing.T) {
	a := assert.New(t)
	ManualSeed(1)
	x := torch.RandN([]int64{1}, false)
	expected := float32(0.66135216)
	a.Equal(expected, x.Item())
}

func TestNormal(t *testing.T) {
	x := torch.Empty([]int64{2, 3}, false)
	Normal(&x, 0.1, 0.2)
	assert.NotNil(t, x.T)
}
