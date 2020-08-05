package gotorch_test

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

type myNet struct {
	L1, L2 torch.Module
}

// MyNet returns a MyNet instance
func MyNet() torch.Module {
	n := &myNet{
		L1: torch.Linear(100, 200, false),
		L2: torch.Linear(200, 10, false),
	}
	return n
}

// Forward executes the calculation
func (n *myNet) Forward(x torch.Tensor) torch.Tensor {
	x = n.L1.Forward(x)
	x = n.L2.Forward(x)
	return x
}

func ExampleSGD() {
	net := MyNet()
	np := torch.GetNamedParameters(net)
	fmt.Println(len(np))

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
	// 2
}
