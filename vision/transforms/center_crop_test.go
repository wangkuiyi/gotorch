package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCenterCrop(t *testing.T) {
	a := assert.New(t)

	blue := color.RGBA{0, 0, 255, 255}
	m := drawImage(image.Rect(0, 0, 2, 2), blue)

	r := CenterCrop(2, 2).Run(m)
	a.Equal(image.Rectangle{Min: image.Point{0, 0}, Max: image.Point{2, 2}}, r.Bounds())
	a.Equal(blue, r.At(0, 0))

	r = CenterCrop(1, 2).Run(m)
	a.Equal(image.Rectangle{Min: image.Point{0, 0}, Max: image.Point{2, 1}}, r.Bounds())

	a.Panics(func() { CenterCrop(3, 1).Run(m) })
	a.Panics(func() { CenterCrop(1, 3).Run(m) })

	// trans := CenterCrop(50, 50)
	// o := trans.Run(m)
	// outImage := o.(image.Image)
	// a.Equal(50, outImage.Bounds().Max.X)
	// startX := (200 - 50) / 2
	// startY := (200 - 50) / 2
	//	a.Equal(i.At(startX, startY), outImage.At(0, 0))

}
