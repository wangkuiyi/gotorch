package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRandomFlip(t *testing.T) {
	a := assert.New(t)
	m := drawRGBWImage()

	o := RandomHorizontalFlip(0 /*never flip*/).Run(m)
	a.True(colorEqual(red, o.At(0, 0)))
	a.True(colorEqual(green, o.At(0, 1)))
	a.True(colorEqual(blue, o.At(1, 0)))
	a.True(colorEqual(white, o.At(1, 1)))

	o = RandomHorizontalFlip(1 /*always flip*/).Run(m)
	a.True(colorEqual(red, o.At(1, 0)))
	a.True(colorEqual(green, o.At(1, 1)))
	a.True(colorEqual(blue, o.At(0, 0)))
	a.True(colorEqual(white, o.At(0, 1)))

	o = RandomVerticalFlip(0 /*never flip*/).Run(m)
	a.True(colorEqual(red, o.At(0, 0)))
	a.True(colorEqual(green, o.At(0, 1)))
	a.True(colorEqual(blue, o.At(1, 0)))
	a.True(colorEqual(white, o.At(1, 1)))

	o = RandomVerticalFlip(1 /*always flip*/).Run(m)
	a.True(colorEqual(red, o.At(0, 1)))
	a.True(colorEqual(green, o.At(0, 0)))
	a.True(colorEqual(blue, o.At(1, 1)))
	a.True(colorEqual(white, o.At(1, 0)))
}
