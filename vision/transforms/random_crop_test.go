package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRandomCrop(t *testing.T) {
	a := assert.New(t)
	m := drawImage(image.Rect(0, 0, 2, 2), blue)

	o := RandomCrop(2, 2).Run(m)
	a.Equal(2, o.Bounds().Max.X)
	a.Equal(2, o.Bounds().Max.Y)
	a.True(colorEqual(blue, o.At(0, 0)))
	a.True(colorEqual(blue, o.At(1, 0)))
	a.True(colorEqual(blue, o.At(0, 1)))
	a.True(colorEqual(blue, o.At(1, 1)))

	o = RandomCrop(1, 2).Run(m)
	a.Equal(2, o.Bounds().Max.X)
	a.Equal(1, o.Bounds().Max.Y)
	a.True(colorEqual(blue, o.At(0, 0)))
	a.True(colorEqual(blue, o.At(1, 0)))

	o = RandomCrop(1, 1).Run(m)
	a.Equal(1, o.Bounds().Max.X)
	a.Equal(1, o.Bounds().Max.Y)
	a.True(colorEqual(blue, o.At(0, 0)))
}

func TestRandomCropWrongSizePanics(t *testing.T) {
	a := assert.New(t)
	m := drawImage(image.Rect(0, 0, 1, 1), blue)
	trans := RandomCrop(1, 2)
	a.Panics(func() {
		trans.Run(m)
	})
}
