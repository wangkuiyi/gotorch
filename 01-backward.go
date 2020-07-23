package main

import (
	"fmt"

	"github.com/gotorch/gotorch/at"
	"github.com/gotorch/gotorch/torch"
)

func main() {
	a := torch::randn([]int{3, 4}, at::TensorOptions().requires_grad(true))
	fmt.Println("a = ", a)

	b := torch::randn([]int{4, 1}, at::TensorOptions().requires_grad(true))
	fmt.Println("b = ", b)

	c := at::mm(a, b)
	fmt.Println("c = ", c)

	d := c.sum()
	fmt.Println("d = ", d)

	d.backward();

	fmt.Println("a.grad = ", a.grad())
	fmt.Println("b.grad = ", b.grad())
}
