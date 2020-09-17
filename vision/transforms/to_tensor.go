package transforms

import (
	"fmt"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
	"gocv.io/x/gocv"
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
	case gocv.Mat:
		size := gocv.GetBlobSize(v)
		n := int64(size.Val1)
		c := int64(size.Val2)
		h := int64(size.Val3)
		w := int64(size.Val4)
		view, err := v.DataPtrFloat32()
		if err != nil {
			panic(err)
		}
		tensor := torch.FromBlob(unsafe.Pointer(&view[0]), torch.Float,
			[]int64{n, c, h, w})
		return tensor.Permute([]int64{0, 2, 3, 1})
	case int:
		return intToTensor(obj.(int))
	default:
		panic(fmt.Sprintf("ToTensorTransformer can not transform the input type: %T", v))
	}
}

func intToTensor(x int) torch.Tensor {
	array := make([]int32, 1)
	array[0] = int32(x)
	return torch.FromBlob(unsafe.Pointer(&array[0]), torch.Int, []int64{1})
}
