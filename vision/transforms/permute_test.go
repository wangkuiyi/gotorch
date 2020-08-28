package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestPermute(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([][]float32{{3, 1}, {2, 4}})
	trans := Permute([]int64{1, 0})
	y := trans.Run(x)
	expected := torch.NewTensor([][]float32{{3, 2}, {1, 4}})
	a.True(torch.Equal(expected, y.(torch.Tensor)))
}
