package transforms

import (
	"fmt"
	"image"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
)

// ToTensorTransformer transforms an image or an interger into a Tensor.  If the
// image is of type image.Gray, the tensor has one channle; otherwise, the
// tensor has three channels (RGB).
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
	switch img.(type) {
	case *image.Gray, *image.Gray16:
		return grayImageToTensor(img)
	}
	return colorImageToTensor(img)
}

func colorImageToTensor(img image.Image) torch.Tensor {
	w, h := img.Bounds().Max.X, img.Bounds().Max.Y
	array := make([]float32, h*w*3) // 3 channels

	// Convert pixels into the HWC format
	const denom = float32(0xffff)
	i := 0
	for x := 0; x < h; x++ {
		for y := 0; y < w; y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			array[i] = float32(r) / denom
			i++
			array[i] = float32(g) / denom
			i++
			array[i] = float32(b) / denom
			i++
		}
	}
	hwc := torch.FromBlob(unsafe.Pointer(&array[0]), torch.Float,
		[]int64{int64(w), int64(h), 3})
	return hwc.Permute([]int64{2, 0, 1})
}

func grayImageToTensor(img image.Image) torch.Tensor {
	w, h := img.Bounds().Max.X, img.Bounds().Max.Y
	array := make([]float32, h*w) // 1 channel

	// Convert pixels into the HWC format
	const denom = float32(0xffff)
	i := 0
	for x := 0; x < h; x++ {
		for y := 0; y < w; y++ {
			r, _, _, _ := img.At(x, y).RGBA()
			array[i] = float32(r) / denom
			i++
		}
	}
	return torch.FromBlob(unsafe.Pointer(&array[0]), torch.Float,
		[]int64{int64(w), int64(h)})
}

func intToTensor(x int) torch.Tensor {
	array := make([]int32, 1)
	array[0] = int32(x)
	return torch.FromBlob(unsafe.Pointer(&array[0]), torch.Int, []int64{1})
}
