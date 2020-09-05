package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestToImage(t *testing.T) {
	rect2x2 := image.Rectangle{image.Point{0, 0}, image.Point{2, 2}}
	red := color.RGBA{0xff, 0, 0, 0xff}
	green := color.RGBA{0, 0xff, 0, 0xff}
	blue := color.RGBA{0, 0, 0xff, 0xff}
	light := color.Gray{0xff}
	dark := color.Gray{0x00}

	{
		x := torch.NewTensor([][]int32{{0, 1}, {1, 0}})
		y := ToImage().Run(x)
		assert.Equal(t, 1, len(y))
		im, ok := y[0].(*image.Gray)
		assert.True(t, ok)
		assert.NotNil(t, im)
		assert.Equal(t, rect2x2, im.Bounds())
	}
	{
		x := torch.NewTensor([][][]int32{
			{{1, 0}, {0, 1}},
			{{0, 1}, {0, 0}},
			{{0, 0}, {1, 0}}})
		y := ToImage().Run(x)
		assert.Equal(t, 1, len(y))
		im, ok := y[0].(*image.RGBA)
		assert.True(t, ok)
		assert.NotNil(t, im)
		assert.Equal(t, rect2x2, im.Bounds())
		assert.Equal(t, red, im.At(0, 0))
		assert.Equal(t, blue, im.At(0, 1))
		assert.Equal(t, green, im.At(1, 0))
		assert.Equal(t, red, im.At(1, 1))
	}
	{
		x := torch.NewTensor([][][]int32{
			{{1, 0}, {0, 1}}})
		y := ToImage().Run(x)
		assert.Equal(t, 1, len(y))
		im, ok := y[0].(*image.Gray)
		assert.True(t, ok)
		assert.NotNil(t, im)
		assert.Equal(t, rect2x2, im.Bounds())
		assert.Equal(t, light, im.At(0, 0))
		assert.Equal(t, dark, im.At(0, 1))
		assert.Equal(t, dark, im.At(1, 0))
		assert.Equal(t, light, im.At(1, 1))
	}
}
