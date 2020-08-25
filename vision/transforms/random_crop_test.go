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
	cropped, err := trans.Run(m)
	a.NoError(err)
	i, ok := cropped.(image.Image)
	if !ok {
		a.Fail("returned image is not of type image.Image")
	}
	a.Equal(100, i.Bounds().Max.X)
	a.Equal(100, i.Bounds().Max.Y)
}

func TestRandomCropSizeError(t *testing.T) {
	a := assert.New(t)

	m := generateRandImage(image.Rect(0, 0, 200, 200))
	trans := RandomCrop(300, 300)

	_, err := trans.Run("some string")
	a.Error(err)
	_, err = trans.Run(m)
	a.Error(err)
}
