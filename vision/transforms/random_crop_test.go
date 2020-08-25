package transforms

import (
	"image"
	"image/color"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func generateRandImage(size image.Rectangle) image.Image {
	r0 := rand.New(rand.NewSource(int64(time.Now().Second())))
	i := image.NewRGBA(size)
	for y := 0; y < size.Max.Y-1; y++ {
		for x := 0; x < size.Max.X-1; x++ {
			a0 := uint8(r0.Float32() * 255)
			rgb0 := uint8(r0.Float32() * 255)
			rgb0 = rgb0 * a0
			i.SetRGBA(x, y, color.RGBA{rgb0, rgb0, rgb0, a0})
		}
	}
	return i
}

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
