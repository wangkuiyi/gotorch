package transforms

import (
	"image"
	"image/color"
	"image/draw"
)

func drawImage(size image.Rectangle, c color.Color) image.Image {
	m := image.NewRGBA(size)
	draw.Draw(m, m.Bounds(), &image.Uniform{c}, image.ZP, draw.Src)
	return m
}

func drawGrayImage(size image.Rectangle, c color.Color) *image.Gray {
	m := image.NewGray(size)
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
