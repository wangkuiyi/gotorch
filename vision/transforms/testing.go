package transforms

import (
	"image"
	"image/color"
	"math/rand"
	"time"
)

func generateRandImage(size image.Rectangle) image.Image {
	r0 := rand.New(rand.NewSource(int64(time.Now().Second())))
	i := image.NewNRGBA(size)
	for y := 0; y < size.Max.Y-1; y++ {
		for x := 0; x < size.Max.X-1; x++ {
			a0 := uint8(r0.Float32() * 255)
			rgb0 := uint8(r0.Float32() * 255)
			rgb0 = rgb0 * a0
			i.Set(x, y, color.RGBA{rgb0, rgb0, rgb0, a0})
		}
	}
	return i
}
