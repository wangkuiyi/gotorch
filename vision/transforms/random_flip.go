package transforms

import (
	"fmt"
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
func (t *RandomFlipTransformer) Run(input interface{}) (interface{}, error) {
	i, ok := input.(image.Image)
	if !ok {
		return nil, fmt.Errorf("input should be image.Image type")
	}
	rand.Seed(time.Now().UnixNano())
	if rand.Float32() >= 0.5 {
		return imaging.FlipH(i), nil
	}
	return i, nil
}
