package transforms

import (
	"image"

	"github.com/disintegration/imaging"
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
func (t *ResizeTransformer) Run(input image.Image) image.Image {
	// TODO(typhoonzero): configure resize resampling function other than
	// Linear.
	return imaging.Resize(input, t.width, t.height, imaging.Linear)
}
