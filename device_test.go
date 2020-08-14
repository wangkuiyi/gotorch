package gotorch_test

import (
	"log"
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestDevice(t *testing.T) {
	a := assert.New(t)
	a.NotPanics(func() {
		var device torch.Device
		if torch.IsCUDAAvailable() {
			log.Println("CUDA is valid")
			device = torch.NewDevice("cuda")
		} else {
			log.Println("No CUDA found; CPU only")
			device = torch.NewDevice("cpu")
		}
		torch.RandN([]int64{2, 3}, false).To(device)
	})
}

func TestPanicDevice(t *testing.T) {
	a := assert.New(t)
	a.Panics(func() {
		torch.NewDevice("unknown")
	}, "TestPanicDevice should panics")
}
