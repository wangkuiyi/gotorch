package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestDevice(t *testing.T) {
	a := assert.New(t)
	var device torch.Device
	if torch.IsCUDAAvailable() {
		device = torch.NewDevice("cuda")
	} else {
		device = torch.NewDevice("cpu")
	}
	a.NotNil(device)
	x := torch.RandN([]int64{2, 3}, false)
	x.To(device)
}

func TestPanicDevice(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("TestPanicDevice should have paniced")
		}
	}()
	torch.NewDevice("unknown")
}
