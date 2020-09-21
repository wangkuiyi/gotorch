package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
)

func TestCenterCrop(t *testing.T) {
	a := assert.New(t)

	{
		blue := color.RGBA{0, 0, 255, 255}
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		rCv := CenterCrop(2, 2).Run(mCv)
		r, _ := rCv.ToImage()
		a.Equal(image.Rectangle{Min: image.Point{0, 0}, Max: image.Point{2, 2}}, r.Bounds())
		a.True(colorEqual(blue, r.At(0, 0)))
		a.True(colorEqual(blue, r.At(0, 1)))
		a.True(colorEqual(blue, r.At(1, 0)))
		a.True(colorEqual(blue, r.At(1, 1)))
	}

	{
		blue := color.RGBA{0, 0, 255, 255}
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		rCv := CenterCrop(1, 2).Run(mCv)
		r, _ := rCv.ToImage()
		a.Equal(image.Rectangle{Min: image.Point{0, 0}, Max: image.Point{2, 1}}, r.Bounds())
		a.True(colorEqual(blue, r.At(0, 0)))
		a.True(colorEqual(blue, r.At(1, 0)))
	}

	{
		blue := color.RGBA{0, 0, 255, 255}
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		rCv := CenterCrop(1, 1).Run(mCv)
		r, _ := rCv.ToImage()
		a.Equal(image.Rectangle{Min: image.Point{0, 0}, Max: image.Point{1, 1}}, r.Bounds())
		a.True(colorEqual(blue, r.At(0, 0)))
	}

	{
		blue := color.RGBA{0, 0, 255, 255}
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		a.Panics(func() { CenterCrop(3, 1).Run(mCv) })
		a.Panics(func() { CenterCrop(1, 3).Run(mCv) })
	}
}
