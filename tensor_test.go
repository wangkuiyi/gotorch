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

func ExampleTensor() {
	t := torch.RandN([]int64{10, 100}, false)
	t.Close()
	t.Close()
	// Output:
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

func TestCastTo(t *testing.T) {
	a := torch.NewTensor([]int64{1, 2})
	b := a.CastTo(torch.Float)
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
}

func TestSave(t *testing.T) {
	file, e := ioutil.TempFile("/tmp", "gotroch")
	assert.NoError(t, e)
	defer os.Remove(file.Name())
	tsave := torch.RandN([]int64{2, 3}, false)
	tsave.Save(file.Name())
	tload := torch.Load(file.Name())

	ss := tsave.Shape()
	ts := tload.Shape()
	assert.EqualValues(t, ss, ts)
	assert.Equal(t, tsave.Dtype(), tload.Dtype())
	assert.Equal(t, tsave.String(), tload.String())
}
