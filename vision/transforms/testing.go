package transforms

import (
	"image"
	"image/color"
	"image/draw"
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

func drawImage(size image.Rectangle, c color.Color) image.Image {
	m := image.NewRGBA(size)
	draw.Draw(m, m.Bounds(), &image.Uniform{c}, image.ZP, draw.Src)
	return m
}

func colorEqual(x, y color.Color) bool {
	r1, b1, g1, a1 := x.RGBA()
	r2, b2, g2, a2 := y.RGBA()
	return r1 == r2 && b1 == b2 && g1 == g2 && a1 == a2
}

var (
	red   = color.RGBA{255, 0, 0, 255}
	green = color.RGBA{0, 255, 0, 255}
	blue  = color.RGBA{0, 0, 255, 255}
	white = color.RGBA{255, 255, 255, 255}
)

// Draw a 2x2 pixel image:
//  red | green
//  blue | white
func drawRGBWImage() image.Image {
	m := draw.Image(image.NewRGBA(image.Rect(0, 0, 2, 2)))
	m.Set(0, 0, red)
	m.Set(0, 1, green)
	m.Set(1, 0, blue)
	m.Set(1, 1, white)
	return m
}
