package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCenterCrop(t *testing.T) {
	a := assert.New(t)

	i := generateRandImage(image.Rect(0, 0, 200, 200))
	trans := CenterCrop(50, 250)
	_, err := trans.Run(100)
	a.Error(err)
	_, err = trans.Run(i)
	a.Error(err)

	trans = CenterCrop(50, 50)
	o, err := trans.Run(i)
	a.NoError(err)
	outImage := o.(image.Image)
	a.Equal(50, outImage.Bounds().Max.X)
	startX := (200 - 50) / 2
	startY := (200 - 50) / 2
	a.Equal(i.At(startX, startY), outImage.At(0, 0))

}
