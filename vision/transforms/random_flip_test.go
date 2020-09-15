package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
)

func TestRandomFlip(t *testing.T) {
	a := assert.New(t)

	{
		m := drawRGBWImage()
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomHorizontalFlip(0 /*never flip*/).Run(mCv)
		o, _ := oCv.ToImage()
		a.True(colorEqual(red, o.At(0, 0)))
		a.True(colorEqual(green, o.At(0, 1)))
		a.True(colorEqual(blue, o.At(1, 0)))
		a.True(colorEqual(white, o.At(1, 1)))
	}
	{
		m := drawRGBWImage()
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomHorizontalFlip(1 /*always flip*/).Run(mCv)
		o, _ := oCv.ToImage()
		a.True(colorEqual(red, o.At(0, 1)))
		a.True(colorEqual(green, o.At(0, 0)))
		a.True(colorEqual(blue, o.At(1, 1)))
		a.True(colorEqual(white, o.At(1, 0)))
	}
	{
		m := drawRGBWImage()
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomVerticalFlip(0 /*never flip*/).Run(mCv)
		o, _ := oCv.ToImage()
		a.True(colorEqual(red, o.At(0, 0)))
		a.True(colorEqual(green, o.At(0, 1)))
		a.True(colorEqual(blue, o.At(1, 0)))
		a.True(colorEqual(white, o.At(1, 1)))
	}
	{
		m := drawRGBWImage()
		mCv, _ := gocv.ImageToMatRGB(m)
		oCv := RandomVerticalFlip(1 /*always flip*/).Run(mCv)
		o, _ := oCv.ToImage()
		a.True(colorEqual(red, o.At(1, 0)))
		a.True(colorEqual(green, o.At(1, 1)))
		a.True(colorEqual(blue, o.At(0, 0)))
		a.True(colorEqual(white, o.At(0, 1)))
	}
}
