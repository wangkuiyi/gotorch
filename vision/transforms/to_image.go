package transforms

import (
	"image"
	"image/color"
	"image/draw"

	torch "github.com/wangkuiyi/gotorch"
)

// ToImageTransformer transforms a tensor with numerical elements into a slice
// of image.Image according to the shape.
//
// For shapes like [h, w], the slice contains a gray image of height h and width
// w.
//
// For shapes like [3, h, w], the slice contains a color image.
//
// For shapes like [1, h, w], the slice contains a gray image.
//
// For shapes like [n, h, w], where n is not 1 or 3, the slice contains n gray
// images.
//
// For shapes like [n, 1, h, w], the slice contains n gray images.
//
// For shapes like [n, 3, h, w], the slice contains n color images.
//
type ToImageTransformer struct{}

// ToImage returns ToImageTransformer
func ToImage() *ToImageTransformer {
	return &ToImageTransformer{}
}

// Run executes the ToImageTransformer and returns a Tensor
func (t *ToImageTransformer) Run(x *torch.Tensor) []image.Image {
	if x.T != nil && x.Dtype() != torch.Float {
		x = x.CastTo(torch.Float) // Convert to float32 tensor.
	}

	var r []image.Image
	s := x.Shape()

	switch {
	case len(s) == 2:
		return append(r, toImage(x, nil, false))
	case len(s) == 3 && s[0] == 1:
		return append(r, toImage(x, []int64{0}, false))
	case len(s) == 3 && s[0] == 3:
		return append(r, toImage(x, nil, true))
	case len(s) == 3 && s[0] != 1 && s[0] != 3:
		for n := int64(0); n < s[0]; n++ {
			r = append(r, toImage(x, []int64{n}, false))
		}
		return r
	case len(s) == 4 && s[1] == 1:
		for n := int64(0); n < s[0]; n++ {
			r = append(r, toImage(x, []int64{n, 0}, false))
		}
		return r
	case len(s) == 4 && s[1] == 3:
		for n := int64(0); n < s[0]; n++ {
			r = append(r, toImage(x, []int64{n}, true))
		}
		return r
	}

	return nil
}

func toImage(x *torch.Tensor, idxPrefix []int64, colored bool) image.Image {
	s := x.Shape()
	d := len(s)
	h := s[d-2]
	w := s[d-1]

	var im draw.Image
	rect := image.Rectangle{image.Point{0, 0}, image.Point{int(w), int(h)}}
	if colored {
		im = image.NewRGBA(rect)
	} else {
		im = image.NewGray(rect)
	}

	for i := int64(0); i < h; i++ {
		for j := int64(0); j < w; j++ {
			if colored {
				var c [3]float32
				for ch := int64(0); ch < 3; ch++ {
					idx := append(idxPrefix, ch, i, j)
					c[ch] = x.Index(idx...).Item().(float32)
				}
				im.Set(int(j), int(i), color.RGBA{
					R: uint8(float32(0xff) * c[0]),
					G: uint8(float32(0xff) * c[1]),
					B: uint8(float32(0xff) * c[2]),
					A: 0xff})
			} else {
				idx := append(idxPrefix, i, j)
				g := x.Index(idx...).Item().(float32)
				im.Set(int(j), int(i), color.Gray{
					uint8(float32(0xff) * g)})
			}
		}
	}
	return im
}
