package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToTensor(t *testing.T) {
	a := assert.New(t)
	// image to Tensor
	m := drawImage(image.Rect(0, 0, 2, 2), color.RGBA{0, 0, 255, 255})
	trans := ToTensor()
	out := trans.Run(m)
	a.Equal(out.Shape(), []int64{3, 2, 2})
	a.Equal("(1,.,.) = \n  0  0\n  0  0\n\n(2,.,.) = \n  0  0\n  0  0\n\n(3,.,.) = \n  1  1\n  1  1\n[ CPUFloatType{3,2,2} ]",
		out.String())

	// int to Tensor
	out = trans.Run(10)
	a.Equal(out.Shape(), []int64{1})
	a.Equal(" 10\n[ CPUIntType{1} ]", out.String())

	// converting any other type to tensor would panic.
	a.Panics(func() {
		trans.Run("hello")
	})
}
