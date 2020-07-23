package main

import (
	"fmt"

	"github.com/gotorch/gotorch/at"
	"github.com/gotorch/gotorch/torch"
)

func main() {
	const (
		N = 64
		D_in = 1000
		H = 100
		D_out = 10
		learning_rate = 1e-6
	)

	x := torch::randn([]int{N, D_in}, at::TensorOptions().requires_grad(false))
	y := torch::randn([]int{N, D_out}, at::TensorOptions().requires_grad(false))

	w1 := torch::randn([]int{D_in, H}, at::TensorOptions().requires_grad(true));
	w2 := torch::randn([]int{H, D_out}, at::TensorOptions().requires_grad(true));

	for i := 0; i < 500; i++ {
		y_pred := at::mm(at::clamp(at::mm(x, w1), 0), w2)
		loss := at::sum(at::pow(at::sub(y_pred, y), 2))

		if i % 100 == 0 {
			fmt.Println("loss =", loss)
		}

		loss.backward()

		at.GradMode.set_enabled(false)
		w1.sub_(w1.grad(), learning_rate)
		w2.sub_(w2.grad(), learning_rate)
		w1.grad().zero_()
		w2.grad().zero_()
		at.GradMode.set_enabled(true)
	}
}
