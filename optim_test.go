package gotorch_test

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
)

type myNet struct {
	L1, L2 nn.Module
}

// MyNet returns a MyNet instance
func MyNet() nn.Module {
	n := &myNet{
		L1: nn.Linear(100, 200, false),
		L2: nn.Linear(200, 10, false),
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
	np := nn.GetNamedParameters(net)
	fmt.Println(len(np))

	opt := torch.SGD(0.1, 0, 0, 0, false)
	opt.AddParameters(nn.GetParameters(net))

	for i := 0; i < 100; i++ {
		torch.GC()
		data := torch.RandN([]int64{32, 100}, false)
		pre := net.Forward(data)
		loss := torch.Sum(pre)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	torch.FinishGC()
	opt.Close()
	nn.CloseModule(net)

	// Output:
	// 2
}
