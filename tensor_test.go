package gotorch_test

import (
	"io/ioutil"
	"os"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

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

func TestClone(t *testing.T) {
	data := []float32{2.0}
	a := torch.FromBlob(unsafe.Pointer(&data[0]), torch.Float, []int64{1})
	b := a.Clone()

	assert.Equal(t, float32(2.0), a.Item().(float32))
	assert.Equal(t, float32(2.0), b.Item().(float32))

	data[0] = 1.0
	assert.Equal(t, float32(1.0), a.Item().(float32))
	assert.Equal(t, float32(2.0), b.Item().(float32))
}

func TestNumel(t *testing.T) {
	x := torch.RandN([]int64{3, 4}, true)
	assert.Equal(t, int64(12), x.Numel())
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

func TestCastTo(t *testing.T) {
	a := torch.NewTensor([]int64{1, 2})
	b := a.CastTo(torch.Float)
	assert.Equal(t, torch.Float, b.Dtype())
	b = a.To(torch.NewDevice("cpu"))
	assert.Equal(t, torch.Long, b.Dtype())
	b = a.To(torch.NewDevice("cpu"), torch.Float)
	assert.Equal(t, torch.Float, b.Dtype())
}

func TestCopyTo(t *testing.T) {
	device := torch.NewDevice("cpu")
	a := torch.NewTensor([]int64{1, 2})
	b := a.CopyTo(device)
	assert.True(t, torch.Equal(a, b))
}

func TestDim(t *testing.T) {
	a := torch.RandN([]int64{2, 3}, false)
	assert.Equal(t, int64(2), a.Dim())
}

func TestShape(t *testing.T) {
	a := torch.RandN([]int64{2, 3}, false)
	assert.Equal(t, int64(2), a.Shape()[0])
	assert.Equal(t, int64(3), a.Shape()[1])
	// a.Argmax returns a 0-dim tensor
	b := a.Argmax()
	assert.Equal(t, 0, len(b.Shape()))
}

func TestSave(t *testing.T) {
	file, e := ioutil.TempFile("", "gotroch-test-save-*")
	assert.NoError(t, e)
	defer os.Remove(file.Name())

	a := torch.RandN([]int64{2, 3}, false)
	a.Save(file.Name())
	b := torch.Load(file.Name())

	assert.EqualValues(t, a.Shape(), b.Shape())
	assert.Equal(t, a.Dtype(), b.Dtype())
	assert.Equal(t, a.String(), b.String())
}

func TestSetData(t *testing.T) {
	a := torch.Full([]int64{2, 3}, 0, false)
	b := torch.Ones([]int64{2, 3}, false)
	assert.False(t, torch.Equal(a, b))
	b.SetData(a)
	assert.True(t, torch.Equal(a, b))
}

func TestTensorIndex(t *testing.T) {
	a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, float32(1), a.Index(0, 0).Item().(float32))
	assert.Equal(t, float32(2), a.Index(0, 1).Item().(float32))
	assert.Equal(t, float32(3), a.Index(1, 0).Item().(float32))
	assert.Equal(t, float32(4), a.Index(1, 1).Item().(float32))

	assert.Panics(t, func() { a.Index(0).Item() })
	assert.Panics(t, func() { a.Index(0, 0, 0).Item() })
}

func TestTensorPinMemory(t *testing.T) {
	a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	b := a.PinMemory()
	if torch.IsCUDAAvailable() {
		assert.Equal(t, " 1  2\n 3  4\n[ CUDAFloatType{2,2} ]", b.String())
	} else {
		assert.Equal(t, " 1  2\n 3  4\n[ CPUFloatType{2,2} ]", b.String())
	}
}
