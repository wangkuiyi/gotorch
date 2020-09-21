package transforms

import (
	"math/rand"

	"gocv.io/x/gocv"
)

// RandomHorizontalFlipTransformer randomly flips an image.
type RandomHorizontalFlipTransformer struct {
	p float32
}

// RandomHorizontalFlip returns the RandomHorizontalFlipTransformer.
func RandomHorizontalFlip(p float32) *RandomHorizontalFlipTransformer {
	return &RandomHorizontalFlipTransformer{p: p}
}

// Run execute the random flip function and returns the flipped image object.
func (hf *RandomHorizontalFlipTransformer) Run(input gocv.Mat) gocv.Mat {
	if rand.Float32() < hf.p {
		gocv.Flip(input, &input, 0)
	}
	return input

}

// RandomVerticalFlipTransformer randomly flips an image.
type RandomVerticalFlipTransformer struct {
	p float32
}

// RandomVerticalFlip returns the RandomVerticalFlipTransformer
func RandomVerticalFlip(p float32) *RandomVerticalFlipTransformer {
	return &RandomVerticalFlipTransformer{p: p}
}

// Run execute the random flip function and returns the flipped image object.
func (hf *RandomVerticalFlipTransformer) Run(input gocv.Mat) gocv.Mat {
	if rand.Float32() < hf.p {
		gocv.Flip(input, &input, 1)
	}
	return input
}
