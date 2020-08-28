package nn

import (
	"log"
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

type myNet struct {
	Module
	L1, L2 *LinearModule
}

func newMyNet() *myNet {
	n := &myNet{
		L1: Linear(100, 200, false),
		L2: Linear(200, 10, false),
	}
	n.Init(n)
	return n
}

// Forward executes the calculation
func (n *myNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	x = n.L2.Forward(x)
	return x
}

type myNetWithBuffer struct {
	Module
	Weight torch.Tensor `gotorch:"buffer"`
	L1     *LinearModule
}

func newMyNetWithBuffer() *myNetWithBuffer {
	n := &myNetWithBuffer{
		Weight: torch.RandN([]int64{100, 200}, false),
		L1:     Linear(100, 200, false),
	}
	n.Init(n)
	return n
}

// Forward executes the calculation
func (n *myNetWithBuffer) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	return x
}

type hierarchicalNet struct {
	Module
	L1 *myNet
	L2 []*LinearModule
}

func newHierarchicalNet() *hierarchicalNet {
	n := &hierarchicalNet{
		L1: newMyNet(),
		L2: []*LinearModule{
			Linear(10, 10, false),
			Linear(10, 5, false),
		},
	}
	n.Init(n)
	return n
}

// Forward executes the calculation
func (n *hierarchicalNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	x = n.L2[0].Forward(x)
	x = n.L2[1].Forward(x)
	return x
}

type myNet3 struct {
	Module
	L1 *SequentialModule
	L2 *LinearModule
}

func newMyNet3() *myNet3 {
	n := &myNet3{
		L1: nil,
		L2: Linear(200, 10, false),
	}
	n.Init(n)
	return n
}

// Forward executes the calculation
func (n *myNet3) Forward(x torch.Tensor) torch.Tensor {
	if n.L1 != nil {
		x = n.L1.Forward(x).(torch.Tensor)
	}
	x = n.L2.Forward(x)
	return x
}

func TestModule(t *testing.T) {
	n := newMyNet()
	n.ZeroGrad()
	namedParams := n.NamedParameters()
	assert.Equal(t, 2, len(namedParams))
	assert.Contains(t, namedParams, "myNet.L1.Weight")
	assert.Contains(t, namedParams, "myNet.L2.Weight")

	n2 := newMyNetWithBuffer()
	namedParams2 := n2.NamedParameters()
	assert.Equal(t, 1, len(namedParams2))
	assert.Contains(t, namedParams2, "myNetWithBuffer.L1.Weight")
	assert.Equal(t, 1, len(n2.Parameters()))
	assert.Equal(t, 1, len(n2.Buffers()))

	hn := newHierarchicalNet()
	hnNamedParams := hn.NamedParameters()
	assert.Equal(t, 4, len(hnNamedParams))
	assert.Contains(t, hnNamedParams, "hierarchicalNet.L1.L1.Weight")
	assert.Contains(t, hnNamedParams, "hierarchicalNet.L1.L2.Weight")
	assert.Contains(t, hnNamedParams, "hierarchicalNet.L2[0].Weight")
	assert.Contains(t, hnNamedParams, "hierarchicalNet.L2[1].Weight")
	assert.Equal(t, 4, len(hn.Parameters()))
	assert.Equal(t, 0, len(hn.Buffers()))
}

func TestModuleTrain(t *testing.T) {
	n := newMyNet()
	n.Train(false)
	assert.False(t, n.IsTraining())
	assert.False(t, n.L1.IsTraining())
	assert.False(t, n.L2.IsTraining())

	hn := newHierarchicalNet()
	hn.Train(false)
	assert.False(t, hn.IsTraining())
	assert.False(t, hn.L2[0].IsTraining())
	assert.False(t, hn.L2[1].IsTraining())
	assert.False(t, hn.L1.IsTraining())
	assert.False(t, hn.L1.L1.IsTraining())
	assert.False(t, hn.L1.L2.IsTraining())

	n3 := newMyNet3()
	n3.Train(false)
	assert.False(t, n3.IsTraining())
	assert.False(t, n3.L2.IsTraining())
}

func TestNewModuleWithoutInit(t *testing.T) {
	newMyNetWithoutInit := func() *hierarchicalNet {
		return &hierarchicalNet{
			L1: newMyNet(),
			L2: []*LinearModule{
				Linear(10, 10, false),
				Linear(10, 5, false),
			},
		}
	}
	n := newMyNetWithoutInit()
	assert.Panics(t, func() { n.Train(true) })
	assert.Panics(t, func() { n.To(torch.NewDevice("cpu")) })
	assert.Panics(t, func() { n.ZeroGrad() })
	assert.Panics(t, func() { n.NamedParameters() })
	assert.Panics(t, func() { n.NamedBuffers() })
}

func TestModuleToDevice(t *testing.T) {
	var device torch.Device
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	hn := newHierarchicalNet()
	assert.NotPanics(t, func() { hn.To(device) })
}

func TestModuleStateDict(t *testing.T) {
	n := newMyNetWithBuffer()
	sd := n.StateDict()
	assert.Equal(t, 2, len(sd))
	assert.Contains(t, sd, "myNetWithBuffer.L1.Weight")
	assert.Contains(t, sd, "myNetWithBuffer.Weight")
}
