package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRandomFlip(t *testing.T) {
	a := assert.New(t)
	i := generateRandImage(image.Rect(0, 0, 200, 200))
	width := i.Bounds().Max.X

	trans := RandomFlip()
	o, err := trans.Run(i)
	a.NoError(err)
	outImage := o.(*image.NRGBA)
	inImage := i.(*image.NRGBA)
	a.True(inImage.At(0, 0) == outImage.At(0, 0) || inImage.At(0, 0) == outImage.At(width-1, 0))
}
