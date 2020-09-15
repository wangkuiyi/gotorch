package transforms

import (
	"image"
	"log"
	"math/rand"

	"gocv.io/x/gocv"
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
func (t *RandomCropTransformer) Run(input gocv.Mat) gocv.Mat {
	if w := input.Cols(); t.width > w {
		log.Panicf("RandomCrop: wanted width %d larger than image width %d", t.width, w)
	}
	if h := input.Rows(); t.height > h {
		log.Panicf("RandomCrop: wanted height %d larger than image height %d", t.height, h)
	}
	x := rand.Intn(input.Cols() - t.width + 1)
	y := rand.Intn(input.Rows() - t.height + 1)

	rect := image.Rectangle{
		Min: image.Point{X: x, Y: y},
		Max: image.Point{X: x + t.width, Y: y + t.height},
	}
	cropped := input.Region(rect)
	defer cropped.Close()
	cropped.CopyTo(&input)
	return input
}
