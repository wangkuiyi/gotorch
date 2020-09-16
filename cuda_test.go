package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestCUDAStreamPanics(t *testing.T) {
	a := assert.New(t)
	var device torch.Device
	if torch.IsCUDAAvailable() {
		device = torch.NewDevice("cpu")
		a.Panics(func() {
			torch.GetCurrentStream(device)
		})
	} else {
		a.NotPanics(func() {
			device = torch.NewDevice("cuda")
		})
	}
}
