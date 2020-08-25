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
func (t *RandomCropTransformer) Run(input image.Image) image.Image {
	if t.width > input.Bounds().Max.X || t.height > input.Bounds().Max.Y {
		panic(fmt.Sprintf("crop size (%d, %d) should be within image size (%d, %d)",
			t.width, t.height, input.Bounds().Max.X, input.Bounds().Max.Y))
	}
	rand.Seed(time.Now().UnixNano())
	x := rand.Intn(input.Bounds().Max.X - t.width)
	y := rand.Intn(input.Bounds().Max.Y - t.height)

	rect := image.Rectangle{
		Min: image.Point{X: x, Y: y},
		Max: image.Point{X: x + t.width, Y: y + t.height},
	}
	croped := imaging.Crop(input, rect)
	return croped
}
