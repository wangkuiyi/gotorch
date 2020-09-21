package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
)

func TestToTensorColorImage(t *testing.T) {
	m := drawImage(image.Rect(0, 0, 2, 3), color.RGBA{0, 0, 255, 255})
	mCv, _ := gocv.ImageToMatRGB(m)
	gocv.CvtColor(mCv, &mCv, gocv.ColorBGRToRGB)
	out := ToTensor().Run(mCv)
	assert.Equal(t, []int64{3, 3, 2}, out.Shape())
	assert.Equal(t,
		"(1,.,.) = \n  0  0\n  0  0\n  0  0\n\n(2,.,.) = \n  0  0\n  0  0\n  0  0\n\n(3,.,.) = \n  1  1\n  1  1\n  1  1\n[ CPUFloatType{3,3,2} ]",
		out.String())
}

func TestToTensorGrayImage(t *testing.T) {
	m := drawGrayImage(image.Rect(0, 0, 2, 3), color.Gray{127})
	mCv, _ := gocv.ImageGrayToMatGray(m)
	out := ToTensor().Run(mCv)
	assert.Equal(t, []int64{3, 2}, out.Shape())
	assert.Equal(t,
		" 0.4980  0.4980\n 0.4980  0.4980\n 0.4980  0.4980\n[ CPUFloatType{3,2} ]",
		out.String())
}

func TestToTensorInteger(t *testing.T) {
	out := ToTensor().Run(10)
	assert.Equal(t, out.Shape(), []int64{1})
	assert.Equal(t, " 10\n[ CPUIntType{1} ]", out.String())
}

func TestToTensorUnknownTypePanics(t *testing.T) {
	assert.Panics(t, func() {
		ToTensor().Run("hello")
	})
}
