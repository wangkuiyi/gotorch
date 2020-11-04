package parallel

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	"testing"
)

type myModelModule struct {
	nn.Module // Every model must derive from Module
}

// Forward executes the calculation
func (m *myModelModule) Forward(x torch.Tensor) torch.Tensor {
	fmt.Println("Forward")
	return torch.Tensor{nil}
}

func myModel() *myModelModule {
	m := &myModelModule{}
	m.Init(m)
	return m
}

func TestDataParallel(t *testing.T) {
	m := myModel()
	// panic: Parallel API needs -DWITH_CUDA on building libcgotorch.so
	assert.Panics(t, func() {
		DataParallel(m, torch.Tensor{nil}, []torch.Device{}, torch.Device{}, 0)
	})
}
