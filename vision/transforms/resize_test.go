package transforms

import (
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
)

func TestResize(t *testing.T) {
	a := assert.New(t)
	img := draw.Image(image.NewRGBA(image.Rect(0, 0, 2, 2)))
	img.Set(0, 0, color.White)
	img.Set(1, 1, color.Black)
	img.Set(0, 1, color.White)
	img.Set(1, 0, color.White)
	imgCv, _ := gocv.ImageToMatRGB(img)

	trans := Resize(4, 4)
	oCv := trans.Run(imgCv)
	o, _ := oCv.ToImage()

	a.True(colorEqual(color.White, o.At(0, 0)))
	a.True(colorEqual(color.Black, o.At(3, 3)))

	// linear interpolation after resize will get color between white and black.
	r, _, _, _ := o.At(2, 2).RGBA()
	a.Less(r, uint32(0xffff))
	a.Greater(r, uint32(0))
}
