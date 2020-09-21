package transforms

import (
	"image"

	"gocv.io/x/gocv"
)

// ResizeTransformer resizes the given image.
type ResizeTransformer struct {
	width, height int
}

// Resize returns a ResizeTransformer.
func Resize(height int, width ...int) *ResizeTransformer {
	w := height
	if len(width) > 0 {
		w = width[0]
	}
	return &ResizeTransformer{width: w, height: height}
}

// Run execute the center crop function and returns the cropped image object.
func (t *ResizeTransformer) Run(input gocv.Mat) gocv.Mat {
	// linear resize
	gocv.Resize(input, &input, image.Point{t.width, t.height}, 0, 0, 1)
	return input
}
