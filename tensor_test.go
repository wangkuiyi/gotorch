package gotorch_test

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

func ExampleTensor() {
	t := torch.RandN([]int64{10, 100}, false)
	t.Close()
	t.Close()
	// Output:
}

func TestTensorItem(t *testing.T) {
	x := torch.RandN([]int64{1}, false)
	r := x.Item()
	assert.NotNil(t, r)
}

func TestTensorDetach(t *testing.T) {
	x := torch.RandN([]int64{1}, true)
	y := x.Detach()
	assert.NotNil(t, y.T)
	initializer.Zeros(&y)
	assert.Equal(t, float32(0.0), x.Item())
}

func TestFromBlob(t *testing.T) {
	data := [2][3]float32{{1.0, 1.1, 1.2}, {2, 3, 4}}
	out := torch.FromBlob(unsafe.Pointer(&data), torch.Float, []int64{2, 3})
	assert.Equal(t, []int64{2, 3}, out.Shape())
}

func TestTensorString(t *testing.T) {
	data := [2][3]float32{{1.0, 1.1, 1.2}, {2, 3, 4}}
	out := torch.FromBlob(unsafe.Pointer(&data), torch.Float, []int64{2, 3})
	g := ` 1.0000  1.1000  1.2000
 2.0000  3.0000  4.0000
[ CPUFloatType{2,3} ]`
	assert.Equal(t, g, out.String())
}

func TestTensorGrad(t *testing.T) {
	a := torch.RandN([]int64{10, 10}, true)
	assert.NotNil(t, a.Grad().T)

	// According to libtorch document https://bit.ly/2QnwHrI, either a
	// tensor that requires grad or not, the grad() method returns a tensor.
	//
	/// This function returns an undefined tensor by default and returns a
	/// defined tensor the first time a call to `backward()` computes
	/// gradients for this Tensor.  The attribute will then contain the
	/// gradients computed and future calls to `backward()` will accumulate
	/// (add) gradients into it.
	b := torch.RandN([]int64{10, 10}, false)
	assert.NotNil(t, b.Grad().T)
}
