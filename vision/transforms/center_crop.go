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
func (t *CenterCropTransformer) Run(input interface{}) (interface{}, error) {
	i, ok := input.(image.Image)
	if !ok {
		return nil, fmt.Errorf("input should be image.Image type")
	}
	if t.width > i.Bounds().Max.X || t.height > i.Bounds().Max.Y {
		return nil, fmt.Errorf("crop size (%d, %d) should be within image size (%d, %d)",
			t.width, t.height, i.Bounds().Max.X, i.Bounds().Max.Y)
	}

	return imaging.CropCenter(i, t.width, t.height), nil
}
