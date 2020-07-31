package main

import (
	"fmt"
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	a := torch.RandN(100, 10, true)
	opt := torch.NewSGDOpt(0.1, 0, 0, 0, false)
	opt.AddParameters([]*torch.Tensor{a})

	for i := 0; i < 10; i++ {
		fmt.Println(i)
		b := torch.RandN(10, 100, false)
		pre := torch.MM(b, a)
		loss := torch.Sum(pre)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()

		runtime.GC()
		runtime.GC()
	}
	runtime.GC()
}
