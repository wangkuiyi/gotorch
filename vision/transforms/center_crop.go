package transforms

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

// CenterCropTransformer crops the center of the image into some size.
type CenterCropTransformer struct {
	width, height int
}

// CenterCrop returns the CenterCropTransformer.
func CenterCrop(height int, width ...int) *CenterCropTransformer {
	w := height
	if len(width) > 0 {
		w = width[0]
	}
	return &CenterCropTransformer{width: w, height: height}
}

// Run execute the center crop function and returns the cropped image object.
func (t *CenterCropTransformer) Run(input gocv.Mat) gocv.Mat {
	if t.width > input.Cols() || t.height > input.Rows() {
		panic(fmt.Sprintf("crop size (%d, %d) should be within image size (%d, %d)",
			t.width, t.height, input.Cols(), input.Rows()))
	}
	cropped := input.Region(image.Rect((input.Cols()-t.width)/2,
		(input.Rows()-t.height)/2,
		(input.Cols()+t.width)/2,
		(input.Rows()+t.height)/2))
	defer cropped.Close()
	cropped.CopyTo(&input)
	return input
}
