package main

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	b := torch.RandN(4, 1, true)
	opt := torch.NewSGDOpt(0.1, 0, 0, 0, false)
	opt.AddParameters([]torch.Tensor{b})

	fmt.Println("Parameter value:")
	fmt.Println(b)

	a := torch.RandN(3, 4, false)
	c := torch.MM(a, b)
	d := torch.Sum(c)

	// clear gradients before backward
	opt.ZeroGrad()
	d.Backward()

	fmt.Println("Gradient value:")
	fmt.Println(b.Grad())

	// update parameters
	opt.Step()

	fmt.Println("Parameter value after updating:")
	fmt.Println(b)
}
