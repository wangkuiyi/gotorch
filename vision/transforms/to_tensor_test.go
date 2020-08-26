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
	m := drawImage(image.Rect(0, 0, 4, 4), color.RGBA{0, 0, 255, 255})
	trans := ToTensor()
	out := trans.Run(m)
	a.Equal(out.Shape(), []int64{3, 4, 4})
	// int to Tensor
	out = trans.Run(10)
	a.Equal(out.Shape(), []int64{1})
	a.Panics(func() {
		trans.Run("hello")
	})
}
