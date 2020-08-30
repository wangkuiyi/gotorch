package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
)

type myModelModule struct {
	Module // Every model must derive from Module

	W torch.Tensor // A parameter
	B torch.Tensor // A parameter
	S torch.Tensor `gotorch:"buffer"` // A buffer

	L1 *LinearModule // Has weight but no bias
	L2 *LinearModule // Has weight and bias

	LL []*LinearModule // Slice of modules
	WW []torch.Tensor  // Slice of parameters
	BB []torch.Tensor  // Slice of parameters
	SS []torch.Tensor  `gotorch:"buffer"` // Slice of buffers
}

// Forward executes the calculation
func (m *myModelModule) Forward(x torch.Tensor) torch.Tensor {
	x = F.Linear(x, m.W, m.B)
	x = m.L1.Forward(x)
	x = m.L2.Forward(x)
	for j := range m.LL {
		x = m.LL[j].Forward(x)
	}
	for j := range m.WW {
		x = F.Linear(x, m.WW[j], m.BB[j])
	}
	return x
}

const cardi = 3

func sqaure() torch.Tensor {
	return torch.RandN([]int64{cardi, cardi}, true)
}

func slim() torch.Tensor {
	return torch.RandN([]int64{cardi}, true)
}

func myModel(init bool) *myModelModule {
	m := &myModelModule{
		W: sqaure(),
		B: slim(),
		/* leave S a nil buffer tensor.*/
		L1: Linear(cardi, cardi, true /*has bias*/),
		L2: Linear(cardi, cardi, false /*no bias*/),
		LL: []*LinearModule{
			Linear(100, 200, true /*has bias*/),
			Linear(100, 200, true /*has bias*/),
		},
		WW: []torch.Tensor{sqaure(), sqaure()},
		BB: []torch.Tensor{slim(), slim()},
		SS: []torch.Tensor{slim(), sqaure()},
	}
	if init {
		m.Init(m)
	}
	return m
}

func TestModulePanicIfNotInit(t *testing.T) {
	m := myModel(false)
	assert.Panics(t, func() { m.Train(true) })
	assert.Panics(t, func() { m.To(torch.NewDevice("cpu")) })
	assert.Panics(t, func() { m.NamedParameters() })
	assert.Panics(t, func() { m.NamedBuffers() })
}

func TestModuleTrain(t *testing.T) {
	m := myModel(true)

	assert.True(t, m.IsTraining())
	assert.True(t, m.L1.IsTraining())
	assert.True(t, m.L2.IsTraining())
	assert.True(t, m.LL[0].IsTraining())
	assert.True(t, m.LL[1].IsTraining())

	m.Train(false)

	assert.False(t, m.IsTraining())
	assert.False(t, m.L1.IsTraining())
	assert.False(t, m.L2.IsTraining())
	assert.False(t, m.LL[0].IsTraining())
	assert.False(t, m.LL[1].IsTraining())
}

// func TestNewModuleWithoutInit(t *testing.T) {
// 	newMyNetWithoutInit := func() *hierarchicalNet {
// 		return &hierarchicalNet{
// 			L1: newMyNet(),
// 			L2: []*LinearModule{
// 				Linear(10, 10, false),
// 				Linear(10, 5, false),
// 			},
// 		}
// 	}
// 	n := newMyNetWithoutInit()
// 	assert.Panics(t, func() { n.Train(true) })
// 	assert.Panics(t, func() { n.To(torch.NewDevice("cpu")) })
// 	assert.Panics(t, func() { n.NamedParameters() })
// 	assert.Panics(t, func() { n.NamedBuffers() })
// }

// func TestModuleToDevice(t *testing.T) {
// 	var device torch.Device
// 	if torch.IsCUDAAvailable() {
// 		log.Println("CUDA is valid")
// 		device = torch.NewDevice("cuda")
// 	} else {
// 		log.Println("No CUDA found; CPU only")
// 		device = torch.NewDevice("cpu")
// 	}

// 	hn := newHierarchicalNet()
// 	assert.NotPanics(t, func() { hn.To(device) })
// }

// func TestModuleStateDict(t *testing.T) {
// 	n := newMyNetWithBuffer()
// 	sd := n.StateDict()
// 	assert.Equal(t, 2, len(sd))
// 	assert.Contains(t, sd, "myNetWithBuffer.L1.Weight")
// 	assert.Contains(t, sd, "myNetWithBuffer.Weight")
// }

// func TestModuleGobStateDict(t *testing.T) {
// 	x := newMyNetWithBuffer()
// 	x.L1.Weight = torch.NewTensor([][]float32{{0, 1}, {1, 0}})
// 	x.Weight = torch.NewTensor([]float32{10, 20})

// 	var buf bytes.Buffer
// 	sd := x.StateDict()
// 	assert.NoError(t, gob.NewEncoder(&buf).Encode(sd))

// 	ns := make(map[string]torch.Tensor)
// 	assert.NoError(t, gob.NewDecoder(&buf).Decode(&ns))

// 	assert.Contains(t, ns, "myNetWithBuffer.L1.Weight")
// 	assert.Contains(t, ns, "myNetWithBuffer.Weight")

// 	assert.Equal(t, sd["myNetWithBuffer.L1.Weight"].String(), ns["myNetWithBuffer.L1.Weight"].String())
// 	assert.Equal(t, sd["myNetWithBuffer.Weight"].String(), ns["myNetWithBuffer.Weight"].String())

// 	assert.Equal(t, " 0  1\n 1  0\n[ CPUFloatType{2,2} ]", sd["myNetWithBuffer.L1.Weight"].String())
// 	assert.Equal(t, " 10\n 20\n[ CPUFloatType{2} ]", sd["myNetWithBuffer.Weight"].String())
// }

// func TestModuleSetStateDict(t *testing.T) {
// 	x := newMyNetWithBuffer()
// 	x.L1.Weight = torch.NewTensor([][]float32{{0, 1}, {1, 0}})
// 	x.Weight = torch.NewTensor([]float32{10, 20})

// 	y := newMyNetWithBuffer()
// 	assert.NoError(t, y.SetStateDict(x.StateDict()))

// 	sd := x.StateDict()
// 	assert.Equal(t, 2, len(sd))
// 	assert.Contains(t, sd, "myNetWithBuffer.L1.Weight")
// 	assert.Contains(t, sd, "myNetWithBuffer.Weight")

// 	ns := y.StateDict()
// 	assert.Equal(t, 2, len(sd))
// 	assert.Contains(t, ns, "myNetWithBuffer.L1.Weight")
// 	assert.Contains(t, ns, "myNetWithBuffer.Weight")

// 	assert.Equal(t, sd["myNetWithBuffer.L1.Weight"].String(), ns["myNetWithBuffer.L1.Weight"].String())
// 	assert.Equal(t, sd["myNetWithBuffer.Weight"].String(), ns["myNetWithBuffer.Weight"].String())

// 	assert.Equal(t, " 0  1\n 1  0\n[ CPUFloatType{2,2} ]", sd["myNetWithBuffer.L1.Weight"].String())
// 	assert.Equal(t, " 10\n 20\n[ CPUFloatType{2} ]", sd["myNetWithBuffer.Weight"].String())
// }
