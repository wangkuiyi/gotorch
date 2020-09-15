package transforms

import (
	"image"
	"math"
	"math/rand"

	"gocv.io/x/gocv"
)

// Port from torch vision transform function:
// https://github.com/pytorch/vision/blob/be8192e20d2529fa552bcfc099974da45365ffd6/torchvision/transforms/transforms.py#L708

// RandomResizedCropTransformer randomly crops a image into some size.
type RandomResizedCropTransformer struct {
	width, height  int
	scale0, scale1 float64
	ratio0, ratio1 float64
	interpolation  gocv.InterpolationFlags
}

// RandomResizedCrop returns the RandomResizedCropTransformer.
func RandomResizedCrop(height int, width ...int) *RandomResizedCropTransformer {
	w := height
	if len(width) > 0 {
		w = width[0]
	}
	return RandomResizedCropD(
		height, w,
		0.08, 1.0,
		3.0/4.0, 4.0/3.0,
		gocv.InterpolationLinear,
	)
}

// RandomResizedCropD returns a RandomResizedCropTransformer with detailed params.
func RandomResizedCropD(height, width int, scale0, scale1, ratio0, ratio1 float64, interpolation gocv.InterpolationFlags) *RandomResizedCropTransformer {
	return &RandomResizedCropTransformer{
		width:         width,
		height:        height,
		scale0:        scale0,
		scale1:        scale1,
		ratio0:        ratio0,
		ratio1:        ratio1,
		interpolation: gocv.InterpolationLinear,
	}
}

func uniform(from, to float64) float64 {
	return (rand.Float64() + from) * (to - from)
}

func (t *RandomResizedCropTransformer) getParams(input gocv.Mat) (int, int, int, int) {
	width := input.Cols()
	height := input.Rows()
	area := width * height

	// try 10 times to generate random scaled image bounds.
	for idx := 0; idx < 10; idx++ {
		targetArea := float64(area) * uniform(t.scale0, t.scale1)
		logRatio0 := math.Log(t.ratio0)
		logRatio1 := math.Log(t.ratio1)
		aspectRatio := math.Exp(uniform(logRatio0, logRatio1))

		w := int(math.Round(math.Sqrt(targetArea * aspectRatio)))
		h := int(math.Round(math.Sqrt(targetArea / aspectRatio)))

		if 0 < w && w <= width && 0 < h && h <= height {
			i := rand.Intn(height - h + 1)
			j := rand.Intn(width - w + 1)
			return i, j, h, w
		}
	}
	// Fallback to central crop
	var i, j, w, h int
	inRatio := float64(width) / float64(height)
	ratioMin := math.Min(t.ratio0, t.ratio1)
	ratioMax := math.Max(t.ratio0, t.ratio1)
	if inRatio < ratioMin {
		w = int(width)
		h = int(math.Round(float64(w) / ratioMin))
	} else if inRatio > ratioMax {
		h = int(height)
		w = int(math.Round(float64(h) * ratioMax))
	} else { // whole image
		w = int(width)
		h = int(height)
	}
	i = (height - h) / 2
	j = (width - w) / 2
	return i, j, h, w
}

// Run execute the random crop function and returns the cropped image object.
func (t *RandomResizedCropTransformer) Run(input gocv.Mat) gocv.Mat {
	i, j, h, w := t.getParams(input)
	cropped := input.Region(image.Rectangle{
		Min: image.Point{X: j, Y: i},
		Max: image.Point{X: j + w, Y: i + h},
	})
	defer cropped.Close()
	gocv.Resize(cropped, &input, image.Point{t.width, t.height}, 0, 0, t.interpolation)
	return input
}
