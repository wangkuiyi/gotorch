package transforms

import (
	"image"
	"math/rand"
	"time"

	"github.com/disintegration/imaging"
)

// RandomFlipTransformer randomly flips an image.
type RandomFlipTransformer struct {
}

// RandomFlip returns the RandomFlipTransformer.
func RandomFlip() *RandomFlipTransformer {
	return &RandomFlipTransformer{}
}

// Run execute the random flip function and returns the flipped image object.
func (t *RandomFlipTransformer) Run(input image.Image) image.Image {
	rand.Seed(time.Now().UnixNano())
	if rand.Float32() >= 0.5 {
		return imaging.FlipH(input)
	}
	return input
}
