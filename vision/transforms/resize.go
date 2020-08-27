package transforms

import (
	"image"

	"github.com/disintegration/imaging"
)

// ResizeTransformer struct
type ResizeTransformer struct {
	height, width int
}

// Resize returns the ResizeTransformer.
func Resize(height int, width ...int) *ResizeTransformer {
	w := height
	if len(width) > 0 {
		w = width[0]
	}
	return &ResizeTransformer{height, w}
}

// Run execute the resize function and returns the resized image object.
func (t *ResizeTransformer) Run(input image.Image) image.Image {
	return imaging.Resize(input, t.width, t.height, imaging.Lanczos)
}
