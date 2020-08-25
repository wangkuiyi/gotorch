package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRandomCrop(t *testing.T) {
	a := assert.New(t)

	m := generateRandImage(image.Rect(0, 0, 200, 200))
	trans := RandomCrop(100, 100)
	cropped := trans.Run(m)
	a.Equal(100, cropped.Bounds().Max.X)
	a.Equal(100, cropped.Bounds().Max.Y)
}

func TestRandomCropSizePanics(t *testing.T) {
	a := assert.New(t)
	m := generateRandImage(image.Rect(0, 0, 200, 200))
	trans := RandomCrop(300, 300)
	a.Panics(func() {
		trans.Run(m)
	})
}
