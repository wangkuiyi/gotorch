package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func getDefaultDevice() torch.Device {
	var device torch.Device
	if torch.IsCUDAAvailable() {
		device = torch.NewDevice("cuda")
	} else {
		device = torch.NewDevice("cpu")
	}
	return device
}
func TestCUDAStreamPanics(t *testing.T) {
	a := assert.New(t)
	device := getDefaultDevice()
	if torch.IsCUDAAvailable() {
		a.NotPanics(func() {
			torch.GetCurrentCUDAStream(device)
		})
	} else {
		a.Panics(func() {
			torch.GetCurrentCUDAStream(device)
		})
		a.Panics(func() {
			torch.NewCUDAStream(device)
		})
	}
}

func TestMultiCUDAStream(t *testing.T) {
	if !torch.IsCUDAAvailable() {
		t.Skip("skip TestMultiCUDAStream which only run on CUDA device")
	}
	a := assert.New(t)
	device := getDefaultDevice()
	currStream := torch.GetCurrentCUDAStream(device)
	defer torch.SetCurrentCUDAStream(currStream)
	// create a new CUDA stream
	stream := torch.NewCUDAStream(device)
	// switch to the new CUDA stream
	torch.SetCurrentCUDAStream(stream)
	// copy Tensor from host to device async
	input := torch.RandN([]int64{100, 200}, true).PinMemory()
	input.CUDA(device, true /**nonBlocking=true**/)
	// wait until all tasks completed
	stream.Synchronize()
	// make sure all tasks completed
	a.True(stream.Query())
}
