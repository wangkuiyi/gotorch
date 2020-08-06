package gotorch_test

import (
	torch "github.com/wangkuiyi/gotorch"
)

func ExampleBackward() {
	b := torch.RandN(4, 1, true)
	opt := torch.SGD(0.1, 0, 0, 0, false)
	opt.AddParameters([]torch.Tensor{b})

	a := torch.RandN(3, 4, false)
	c := torch.MM(a, b)
	d := torch.Sum(c)

	// clear gradients before backward
	opt.ZeroGrad()
	d.Backward()

	// update parameters
	opt.Step()

	//Output:
}
