package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestDeviceTo(t *testing.T) {
	a := assert.New(t)
	a.NotPanics(func() {
		var device torch.Device
		if torch.IsCUDAAvailable() {
			t.Log("CUDA is valid")
			device = torch.NewDevice("cuda")
		} else {
			t.Log("No CUDA found; CPU only")
			device = torch.NewDevice("cpu")
		}
		torch.RandN([]int64{2, 3}, false).To(device, torch.Float)
	})
}

func TestDevicePanicWithUnknown(t *testing.T) {
	a := assert.New(t)
	a.Panics(func() {
		torch.NewDevice("unknown")
	}, "TestPanicDevice should panics")
}

func TestDeviceIsCUDNNAvailable(t *testing.T) {
	if torch.IsCUDNNAvailable() {
		t.Log("CUDNN is available")
	} else {
		t.Log("No CUDNN found")
	}
}
