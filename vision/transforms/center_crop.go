package transforms

import (
	"fmt"
	"image"

	"github.com/disintegration/imaging"
)

// CenterCropTransformer crops the center of the image into some size.
type CenterCropTransformer struct {
	width, height int
}

// CenterCrop returns the CenterCropTransformer.
func CenterCrop(width, height int) *CenterCropTransformer {
	return &CenterCropTransformer{width, height}
}

// Run execute the center crop function and returns the cropped image object.
func (t *CenterCropTransformer) Run(input image.Image) image.Image {
	if t.width > input.Bounds().Max.X || t.height > input.Bounds().Max.Y {
		panic(fmt.Sprintf("crop size (%d, %d) should be within image size (%d, %d)",
			t.width, t.height, input.Bounds().Max.X, input.Bounds().Max.Y))
	}

	return imaging.CropCenter(input, t.width, t.height)
}
