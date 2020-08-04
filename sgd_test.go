package gotorch_test

import (
	torch "github.com/wangkuiyi/gotorch"
)

type myNet struct {
	torch.Model
	l1, l2 torch.Module
}

// MyNet returns a MyNet instance
func MyNet() torch.Module {
	n := &myNet{
		l1: torch.Linear(100, 200, false),
		l2: torch.Linear(200, 10, false),
	}
	n.RegisterModule("l1", n.l1)
	n.RegisterModule("l2", n.l2)
	return n
}

// Forward executes the calculation
func (n *myNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.l1.Forward(x)
	x = n.l2.Forward(x)
	return x
}

func ExampleSGD() {
	net := MyNet()
	opt := torch.SGD(0.1, 0, 0, 0, false)
	opt.AddParameters(torch.GetParameters(net))

	for i := 0; i < 100; i++ {
		torch.GC()
		data := torch.RandN(32, 100, false)
		pre := net.Forward(data)
		loss := torch.Sum(pre)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	torch.FinishGC()
	opt.Close()
	torch.CloseModule(net)
	// Output:
}
