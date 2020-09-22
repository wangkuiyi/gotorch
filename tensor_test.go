package gotorch_test

import (
	"io/ioutil"
	"os"
	"runtime"
	"testing"
	"time"
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
func TestCUDA(t *testing.T) {
	a := assert.New(t)
	device := getDefaultDevice()
	input := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	if !torch.IsCUDAAvailable() {
		// CUDA should panics on CPU device
		a.Panics(func() {
			input.CUDA(device, false)
		})
		a.Panics(func() {
			input.CUDA(device, true)
		})
		return
	}

	b := input.CUDA(device, false)
	a.Equal(" 1  2\n 3  4\n[ CUDAFloatType{2,2} ]", b.String())

	c := input.CUDA(device, true)
	torch.GetCurrentCUDAStream(device).Synchronize()
	a.Equal(" 1  2\n 3  4\n[ CUDAFloatType{2,2} ]", c.String())
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

func TestTensorGC(t *testing.T) {
	torch.GC()
	defer torch.FinishGC()
	runtime.LockOSThread()
	c := make(chan torch.Tensor, 0)
	{
		torch.NewTensor([][]float32{{1, 2}, {3, 4}})
		go func() {
			a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
			c <- a
			time.Sleep(time.Second)
			runtime.KeepAlive(&a)
		}()
	}
	<-c
	assert.Eventually(t, func() bool { torch.GC(); return true }, 10*time.Millisecond, 10*time.Microsecond)
}
