package transforms

import (
	"image"
	"image/draw"
	"math/rand"
	"testing"

	"github.com/disintegration/imaging"
	"github.com/stretchr/testify/assert"
)

func TestRandomResizedCrop(t *testing.T) {
	a := assert.New(t)

	input := image.NewRGBA(image.Rectangle{
		Min: image.Point{X: 0, Y: 0},
		Max: image.Point{X: 50, Y: 50},
	})
	draw.Draw(input, input.Bounds(), &image.Uniform{blue}, image.ZP, draw.Src)
	input.Set(19, 19, red)

	// Test crop output original size and color.
	trans := RandomResizedCropD(50, 50, 1.0, 1.0, 1.0, 1.0, imaging.Linear)
	out := trans.Run(input)
	a.Equal(50, out.Bounds().Max.X)
	a.Equal(50, out.Bounds().Max.Y)
	a.True(colorEqual(red, out.At(19, 19)))

	// bounds test for scalar and ratio equals to zero
	trans = RandomResizedCropD(50, 50, 0.0, 0.0, 0.0, 0.0, imaging.Linear)
	out = trans.Run(input)
	a.Equal(0, out.Bounds().Max.X)
	a.Equal(0, out.Bounds().Max.Y)

	rand.Seed(1)
	// Test crop output smaller size
	trans = RandomResizedCrop(20)
	out = trans.Run(input)
	a.Equal(20, out.Bounds().Max.X)
	r, g, b, _ := out.At(7, 5).RGBA()
	// the resized image color should be between blue and red
	a.True(r > uint32(0x0) && r < uint32(0xffff))
	a.Equal(uint32(0x0000), g)
	a.True(b > uint32(0x0) && b < uint32(0xffff))

	// Test crop output greater size
	trans = RandomResizedCrop(60, 80)
	out = trans.Run(input)
	a.Equal(80, out.Bounds().Max.X)
	a.Equal(60, out.Bounds().Max.Y)
}
