package transforms

import (
	"fmt"
	"image"
	"math/rand"
	"time"

	"github.com/disintegration/imaging"
)

// RandomCropTransformer randomly crops a image into some size.
type RandomCropTransformer struct {
	width, height int
}

// RandomCrop returns the RandomCropTransformer.
func RandomCrop(width, height int) *RandomCropTransformer {
	return &RandomCropTransformer{width, height}
}

// Run execute the random crop function and returns the cropped image object.
func (t *RandomCropTransformer) Run(input interface{}) (interface{}, error) {
	i, ok := input.(image.Image)
	if !ok {
		return nil, fmt.Errorf("input should be image.Image type")
	}
	if t.width > i.Bounds().Max.X || t.height > i.Bounds().Max.Y {
		return nil, fmt.Errorf("crop size (%d, %d) should be within image size (%d, %d)",
			t.width, t.height, i.Bounds().Max.X, i.Bounds().Max.Y)
	}
	rand.Seed(time.Now().UnixNano())
	x := rand.Intn(i.Bounds().Max.X - t.width)
	y := rand.Intn(i.Bounds().Max.Y - t.height)

	rect := image.Rectangle{
		Min: image.Point{X: x, Y: y},
		Max: image.Point{X: x + t.width, Y: y + t.height},
	}
	croped := imaging.Crop(i, rect)
	return croped, nil
}
