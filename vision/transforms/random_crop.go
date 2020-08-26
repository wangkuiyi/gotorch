package transforms

import (
	"image"
	"log"
	"math/rand"

	"github.com/disintegration/imaging"
)

// RandomCropTransformer randomly crops a image into some size.
type RandomCropTransformer struct {
	width, height int
}

// RandomCrop returns the RandomCropTransformer.
func RandomCrop(height int, width ...int) *RandomCropTransformer {
	w := height
	if len(width) > 0 {
		w = width[0]
	}
	return &RandomCropTransformer{width: w, height: height}
}

// Run execute the random crop function and returns the cropped image object.
func (t *RandomCropTransformer) Run(input image.Image) image.Image {
	if w := input.Bounds().Max.X; t.width > w {
		log.Panicf("RandomCrop: wanted width %d larger than image width %d", t.width, w)
	}
	if h := input.Bounds().Max.Y; t.height > h {
		log.Panicf("RandomCrop: wanted height %d larger than image height %d", t.height, h)
	}
	x := rand.Intn(input.Bounds().Max.X - t.width + 1)
	y := rand.Intn(input.Bounds().Max.Y - t.height + 1)

	rect := image.Rectangle{
		Min: image.Point{X: x, Y: y},
		Max: image.Point{X: x + t.width, Y: y + t.height},
	}
	croped := imaging.Crop(input, rect)
	return croped
}
