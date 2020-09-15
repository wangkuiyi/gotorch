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
		c := v.Channels()
		if c == 3 {
			v.ConvertTo(&v, gocv.MatTypeCV32FC3)
		} else {
			v.ConvertTo(&v, gocv.MatTypeCV32FC1)
		}
		v.MultiplyFloat(1.0 / 255.0)

		w := v.Cols()
		h := v.Rows()

		view, err := v.DataPtrFloat32()
		if err != nil {
			panic(err)
		}

		if c == 3 {
			tensor := torch.FromBlob(unsafe.Pointer(&view[0]), torch.Float, []int64{int64(h),
				int64(w), int64(c)})
			return tensor.Permute([]int64{2, 0, 1})
		}
		tensor := torch.FromBlob(unsafe.Pointer(&view[0]), torch.Float, []int64{int64(h),
			int64(w)})
		return tensor
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
