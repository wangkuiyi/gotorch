package nn

import (
	"bytes"
	"encoding/gob"
	"log"
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

func square() torch.Tensor {
	return torch.RandN([]int64{cardi, cardi}, true)
}

func slim() torch.Tensor {
	return torch.RandN([]int64{cardi}, true)
}

func myModel(init bool) *myModelModule {
	m := &myModelModule{
		W: square(),
		B: slim(),
		/* leave S a nil buffer tensor.*/
		L1: Linear(cardi, cardi, true /*has bias*/),
		L2: Linear(cardi, cardi, false /*no bias*/),
		LL: []*LinearModule{
			Linear(100, 200, true /*has bias*/),
			Linear(100, 200, true /*has bias*/),
		},
		WW: []torch.Tensor{square(), square()},
		BB: []torch.Tensor{slim(), slim()},
		SS: []torch.Tensor{slim(), square()},
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
	assert.Panics(t, func() { m.To(torch.NewDevice("cpu"), torch.Int) })
	assert.Panics(t, func() { m.NamedParameters() })
	assert.Panics(t, func() { m.NamedBuffers() })
	assert.Panics(t, func() { m.Parameters() })
	assert.Panics(t, func() { m.Buffers() })
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

func TestModuleToDevice(t *testing.T) {
	var device torch.Device
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	hn := myModel(true)
	assert.NotPanics(t, func() { hn.To(device) })
}

func TestModuleName(t *testing.T) {
	hn := myModel(true)
	assert.Equal(t, "nn.myModelModule", hn.Name())
	assert.Equal(t, "nn.LinearModule", hn.L1.Name())
	assert.Equal(t, "nn.LinearModule", hn.L2.Name())
	assert.Equal(t, "nn.LinearModule", hn.LL[0].Name())
}

func TestModuleOuter(t *testing.T) {
	hn := myModel(true)
	assert.Equal(t, hn, hn.Outer())

	assert.Equal(t, hn.L1, hn.L1.Outer())
	assert.Equal(t, hn.L2, hn.L2.Outer())
	assert.Equal(t, hn.LL[0], hn.LL[0].Outer())
}

func TestModuleStateDict(t *testing.T) {
	n := myModel(true)
	sd := n.StateDict()
	assert.Equal(t, 15, len(sd)) // S is nil
	assert.NotContains(t, sd, "myModelModule.S")
	assert.Contains(t, sd, "myModelModule.W")
	assert.Contains(t, sd, "myModelModule.B")
	assert.Contains(t, sd, "myModelModule.L1.Bias")
	assert.Contains(t, sd, "myModelModule.L1.Weight")
	assert.NotContains(t, sd, "myModelModule.L2.Bias")
	assert.Contains(t, sd, "myModelModule.L2.Weight")
	assert.Contains(t, sd, "myModelModule.LL[0].Bias")
	assert.Contains(t, sd, "myModelModule.LL[0].Weight")
	assert.Contains(t, sd, "myModelModule.LL[1].Bias")
	assert.Contains(t, sd, "myModelModule.LL[1].Weight")
	assert.Contains(t, sd, "myModelModule.WW[0]")
	assert.Contains(t, sd, "myModelModule.WW[1]")
	assert.Contains(t, sd, "myModelModule.BB[0]")
	assert.Contains(t, sd, "myModelModule.BB[1]")
	assert.Contains(t, sd, "myModelModule.SS[0]")
	assert.Contains(t, sd, "myModelModule.SS[1]")
	assert.Equal(t, 13, len(n.Parameters()))
	assert.Equal(t, 2, len(n.Buffers()))
}

func TestModuleGobStateDict(t *testing.T) {
	x := myModel(true)
	x.L1.Weight = torch.NewTensor([][]float32{{0, 1}, {1, 0}})
	x.W = torch.NewTensor([]float32{10, 20})

	var buf bytes.Buffer
	sd := x.StateDict()
	assert.NoError(t, gob.NewEncoder(&buf).Encode(sd))

	ns := make(map[string]torch.Tensor)
	assert.NoError(t, gob.NewDecoder(&buf).Decode(&ns))

	assert.Contains(t, ns, "myModelModule.L1.Weight")
	assert.Contains(t, ns, "myModelModule.W")

	assert.Equal(t, sd["myModelModule.L1.Weight"].String(), ns["myModelModule.L1.Weight"].String())
	assert.Equal(t, sd["myModelModule.W"].String(), ns["myModelModule.W"].String())

	assert.Equal(t, " 0  1\n 1  0\n[ CPUFloatType{2,2} ]", sd["myModelModule.L1.Weight"].String())
	assert.Equal(t, " 10\n 20\n[ CPUFloatType{2} ]", sd["myModelModule.W"].String())
}

func TestModuleSetStateDict(t *testing.T) {
	x := myModel(true)
	x.L1.Weight = torch.NewTensor([][]float32{{0, 1}, {1, 0}})
	x.W = torch.NewTensor([]float32{10, 20})

	y := myModel(true)
	assert.NoError(t, y.SetStateDict(x.StateDict()))

	sd := x.StateDict()
	assert.Equal(t, 15, len(sd))
	assert.Contains(t, sd, "myModelModule.L1.Weight")
	assert.Contains(t, sd, "myModelModule.W")

	ns := y.StateDict()
	assert.Equal(t, 15, len(sd))
	assert.Contains(t, ns, "myModelModule.L1.Weight")
	assert.Contains(t, ns, "myModelModule.W")

	assert.Equal(t, sd["myModelModule.L1.Weight"].String(), ns["myModelModule.L1.Weight"].String())
	assert.Equal(t, sd["myModelModule.W"].String(), ns["myModelModule.W"].String())

	assert.Equal(t, " 0  1\n 1  0\n[ CPUFloatType{2,2} ]", sd["myModelModule.L1.Weight"].String())
	assert.Equal(t, " 10\n 20\n[ CPUFloatType{2} ]", sd["myModelModule.W"].String())
}
