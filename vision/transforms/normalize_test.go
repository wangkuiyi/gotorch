package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestNormalizeTransform(t *testing.T) {
	a := assert.New(t)
	trans := Normalize([]float64{10.0}, []float64{2.3})
	t1 := torch.NewTensor([]float64{10.2, 11.3, 9.2, 8.4, 13.2})
	t2 := trans.Run(t1)

	expected := torch.NewTensor([]float64{
		(10.2 - 10.0) / 2.3,
		(11.3 - 10.0) / 2.3,
		(9.2 - 10.0) / 2.3,
		(8.4 - 10.0) / 2.3,
		(13.2 - 10.0) / 2.3})
	a.True(torch.AllClose(t2.(torch.Tensor), expected))

	a.Panics(func() {
		trans.Run(&t1)
	})
}
