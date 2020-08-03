package main

import (
	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	a := torch.RandN(100, 10, true)
	opt := torch.NewSGDOpt(0.1, 0, 0, 0, false)
	opt.AddParameters([]torch.Tensor{a})

	torch.PrepareGC()
	for i := 0; i < 100; i++ {
		torch.GC()
		b := torch.RandN(10, 100, false)
		pre := torch.MM(b, a)
		loss := torch.Sum(pre)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	torch.FinishGC()
}
