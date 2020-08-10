package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

type myNet struct {
	L1, L2 Module
}

// MyNet returns a myNet instance
func MyNet() Module {
	n := &myNet{
		L1: Linear(100, 200, false),
		L2: Linear(200, 10, false),
	}
	return n
}

// Forward executes the calculation
func (n *myNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	x = n.L2.Forward(x)
	return x
}

type myNet2 struct {
	Weight torch.Tensor `gotorch:"buffer"`
	L1     Module
}

// MyNet2 returns a myNet2 instance
func MyNet2() Module {
	n := &myNet2{
		Weight: torch.RandN([]int{100, 200}, false),
		L1:     Linear(100, 200, false),
	}
	return n
}

// Forward executes the calculation
func (n *myNet2) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	return x
}

type hierarchyNet struct {
	L1, L2 Module
}

// HierarchyNet returns a hierarchyNet instance
func HierarchyNet() Module {
	n := &hierarchyNet{
		L1: MyNet(),
		L2: Linear(200, 10, false),
	}
	return n
}

// Forward executes the calculation
func (n *hierarchyNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	x = n.L2.Forward(x)
	return x
}

func TestModule(t *testing.T) {
	n := MyNet()
	namedParams := GetNamedParameters(n)
	assert.Equal(t, 2, len(namedParams))
	assert.Contains(t, namedParams, "myNet.L1.Weight")
	assert.Contains(t, namedParams, "myNet.L2.Weight")

	n2 := MyNet2()
	namedParams2 := GetNamedParameters(n2)
	assert.Equal(t, 1, len(namedParams2))
	assert.Contains(t, namedParams2, "myNet2.L1.Weight")

	hn := HierarchyNet()
	hnNamedParams := GetNamedParameters(hn)
	assert.Equal(t, 3, len(hnNamedParams))
	assert.Contains(t, hnNamedParams, "hierarchyNet.L1.L1.Weight")
	assert.Contains(t, hnNamedParams, "hierarchyNet.L1.L2.Weight")
	assert.Contains(t, hnNamedParams, "hierarchyNet.L2.Weight")
}

func TestConv2d(t *testing.T) {
	c := Conv2d(16, 33, 3, 2, 0, 1, 1, true, "zeros")
	x := torch.RandN([]int{20, 16, 50, 100}, false)
	output := c.Forward(x)
	assert.NotNil(t, output)
}
