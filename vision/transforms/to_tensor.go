package transforms

import (
	"fmt"
	"image"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
)

// ToTensorTransformer transforms an object to Tensor
type ToTensorTransformer struct{}

// ToTensor returns ToTensorTransformer
func ToTensor() *ToTensorTransformer {
	return &ToTensorTransformer{}
}

// Run executes the ToTensorTransformer and returns a Tensor
func (t ToTensorTransformer) Run(obj interface{}) torch.Tensor {
	switch v := obj.(type) {
	case image.Image:
		return imageToTensor(obj.(image.Image))
	case int:
		return intToTensor(obj.(int))
	default:
		panic(fmt.Sprintf("ToTensorTransformer can not transform the input type: %T", v))
	}
}

// ToTensor transform c.f. https://github.com/pytorch/vision/blob/ba1b22125723f3719a3c38a2fe7cd6fb77657c57/torchvision/transforms/functional.py#L45
func imageToTensor(img image.Image) torch.Tensor {
	width, height := img.Bounds().Max.X, img.Bounds().Max.Y
	// put pixel values with HWC format
	array := make([][][3]float32, height)

	for x := 0; x < height; x++ {
		row := make([][3]float32, width)
		for y := 0; y < width; y++ {
			// ResNet need the 3 channels image, here we should convert to RGB format.
			// The division by 255.0 is applied to convert RGB pixel values from [0, 255] to [0.0, 1.0] range
			switch img := img.(type) {
			case *image.NRGBA:
				c := img.NRGBAAt(x, y)
				row[y] = [3]float32{float32(c.R) / 255.0, float32(c.G) / 255.0, float32(c.B) / 255.0}
			case *image.RGBA:
				c := img.RGBAAt(x, y)
				row[y] = [3]float32{float32(c.R) / 255.0, float32(c.G) / 255.0, float32(c.B) / 255.0}
			}
		}
		array[x] = row
	}
	return torch.FromBlob(unsafe.Pointer(&array[0][0][0]), torch.Float, []int64{int64(width), int64(height), 3})
}

func intToTensor(x int) torch.Tensor {
	array := make([]int, 1)
	array[0] = x
	return torch.FromBlob(unsafe.Pointer(&array[0]), torch.Int, []int64{1})
}
