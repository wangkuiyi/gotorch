package transforms

import (
	"image"
	"math/rand"

	"github.com/disintegration/imaging"
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
func (hf *RandomHorizontalFlipTransformer) Run(input image.Image) image.Image {
	if rand.Float32() < hf.p {
		return imaging.FlipH(input)
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
func (hf *RandomVerticalFlipTransformer) Run(input image.Image) image.Image {
	if rand.Float32() < hf.p {
		return imaging.FlipV(input)
	}
	return input
}
