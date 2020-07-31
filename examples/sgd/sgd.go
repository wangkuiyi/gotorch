package main

import (
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
)

// MyNet struct
type myNet struct {
	l1, l2 torch.Module
}

// MyNet returns a MyNet instance
func MyNet(m *torch.Model) torch.Module {
	return &myNet{
		l1: torch.Linear(m, 100, 200, false),
		l2: torch.Linear(m, 200, 10, false),
	}
}

// Forward executes the calculation
func (n *myNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.l1.Forward(x)
	x = n.l2.Forward(x)
	return x
}

func main() {
	model := torch.NewModel()
	net := MyNet(model)
	opt := torch.NewSGDOpt(0.1, 0, 0, 0, false)
	opt.AddParameters(model.Parameters)

	for i := 0; i < 10; i++ {
		data := torch.RandN(32, 100, false)
		pre := net.Forward(data)
		loss := torch.Sum(pre)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()

		runtime.GC()
	}
	runtime.GC()
}
