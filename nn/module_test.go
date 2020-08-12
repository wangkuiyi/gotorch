package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

type myNet struct {
	Module
	L1, L2 *Linear
}

func newMyNet() *myNet {
	n := &myNet{
		L1: NewLinear(100, 200, false),
		L2: NewLinear(200, 10, false),
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

type myNet2 struct {
	Module
	Weight torch.Tensor `gotorch:"buffer"`
	L1     *Linear
}

func newMyNet2() *myNet2 {
	n := &myNet2{
		Weight: torch.RandN([]int64{100, 200}, false),
		L1:     NewLinear(100, 200, false),
	}
	n.Init(n)
	return n
}

// Forward executes the calculation
func (n *myNet2) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	return x
}

type hierarchyNet struct {
	Module
	L1 *myNet
	L2 *Linear
}

func newHierarchyNet() *hierarchyNet {
	n := &hierarchyNet{
		L1: newMyNet(),
		L2: NewLinear(200, 10, false),
	}
	n.Init(n)
	return n
}

// Forward executes the calculation
func (n *hierarchyNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
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

	n2 := newMyNet2()
	namedParams2 := n2.NamedParameters()
	assert.Equal(t, 1, len(namedParams2))
	assert.Contains(t, namedParams2, "myNet2.L1.Weight")

	hn := newHierarchyNet()
	hnNamedParams := hn.NamedParameters()
	assert.Equal(t, 3, len(hnNamedParams))
	assert.Contains(t, hnNamedParams, "hierarchyNet.L1.L1.Weight")
	assert.Contains(t, hnNamedParams, "hierarchyNet.L1.L2.Weight")
	assert.Contains(t, hnNamedParams, "hierarchyNet.L2.Weight")
}

func TestConv2d(t *testing.T) {
	c := NewConv2d(16, 33, 3, 2, 0, 1, 1, true, "zeros")
	x := torch.RandN([]int64{20, 16, 50, 100}, false)
	output := c.Forward(x)
	assert.NotNil(t, output)
}
