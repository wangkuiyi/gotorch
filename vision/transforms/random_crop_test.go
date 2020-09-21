package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
)

func TestRandomCrop(t *testing.T) {
	a := assert.New(t)

	{
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomCrop(2, 2).Run(mCv)
		o, _ := oCv.ToImage()
		a.Equal(2, o.Bounds().Max.X)
		a.Equal(2, o.Bounds().Max.Y)
		a.True(colorEqual(blue, o.At(0, 0)))
		a.True(colorEqual(blue, o.At(1, 0)))
		a.True(colorEqual(blue, o.At(0, 1)))
		a.True(colorEqual(blue, o.At(1, 1)))
	}

	{
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomCrop(1, 2).Run(mCv)
		o, _ := oCv.ToImage()
		a.Equal(2, o.Bounds().Max.X)
		a.Equal(1, o.Bounds().Max.Y)
		a.True(colorEqual(blue, o.At(0, 0)))
		a.True(colorEqual(blue, o.At(1, 0)))
	}

	{
		m := drawImage(image.Rect(0, 0, 2, 2), blue)
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomCrop(1, 1).Run(mCv)
		o, _ := oCv.ToImage()
		a.Equal(1, o.Bounds().Max.X)
		a.Equal(1, o.Bounds().Max.Y)
		a.True(colorEqual(blue, o.At(0, 0)))
	}
}

func TestRandomCropWrongSizePanics(t *testing.T) {
	a := assert.New(t)
	m := drawImage(image.Rect(0, 0, 1, 1), blue)
	mCv, _ := gocv.ImageToMatRGB(m)
	trans := RandomCrop(1, 2)
	a.Panics(func() {
		trans.Run(mCv)
	})
}
