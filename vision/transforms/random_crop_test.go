package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	blue = color.RGBA{0, 0, 255, 255}
)

func TestRandomCrop(t *testing.T) {
	a := assert.New(t)
	m := drawImage(image.Rect(0, 0, 2, 2), blue)
	trans := RandomCrop(1, 2)
	cropped := trans.Run(m)
	a.Equal(2, cropped.Bounds().Max.X)
	a.Equal(1, cropped.Bounds().Max.Y)

	a.True(colorEqual(blue, cropped.At(0, 0)))
	a.True(colorEqual(blue, cropped.At(1, 0)))
}

func TestRandomCropWrongSizePanics(t *testing.T) {
	a := assert.New(t)
	m := drawImage(image.Rect(0, 0, 1, 1), blue)
	trans := RandomCrop(1, 2)
	a.Panics(func() {
		trans.Run(m)
	})
}
