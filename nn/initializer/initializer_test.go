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
