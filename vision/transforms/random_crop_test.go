package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRandomCrop(t *testing.T) {
	a := assert.New(t)

	m := drawImage(image.Rect(0, 0, 2, 2), color.RGBA{0, 0, 255, 255})
	trans := RandomCrop(1, 2)
	cropped := trans.Run(m)
	a.Equal(2, cropped.Bounds().Max.X)
	a.Equal(1, cropped.Bounds().Max.Y)
}

func TestRandomCropSizePanics(t *testing.T) {
	a := assert.New(t)
	m := generateRandImage(image.Rect(0, 0, 200, 200))
	trans := RandomCrop(300, 300)
	a.Panics(func() {
		trans.Run(m)
	})
}
