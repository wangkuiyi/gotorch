package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCenterCrop(t *testing.T) {
	a := assert.New(t)

	i := generateRandImage(image.Rect(0, 0, 200, 200))
	a.Panics(func() {
		trans := CenterCrop(50, 250)
		trans.Run(i)
	})

	trans := CenterCrop(50, 50)
	o := trans.Run(i)
	outImage := o.(image.Image)
	a.Equal(50, outImage.Bounds().Max.X)
	startX := (200 - 50) / 2
	startY := (200 - 50) / 2
	a.Equal(i.At(startX, startY), outImage.At(0, 0))

}
