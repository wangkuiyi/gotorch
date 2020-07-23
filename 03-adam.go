package main

import (
	"fmt"

	at "github.com/gotorch/gotorch/aten"
	"github.com/gotorch/gotorch/torch"
	"github.com/gotorch/gotorch/torch/optim"
)

func main() {
	N, D_in, H, D_out := 64, 1000, 100, 10
	learning_rate := 1e-3

	x := torch.RandN([]int{N, Din},
		at.TensorOptions().RequiresGrad(false))
	y := torch.RandN([]int{N, Dout},
		at.TensorOptions().RequiresGrad(false))

	params := []at.Tensor{
		torch.RandN([]int{Din, H},
			at.TensorOptions().RequiresGrad(true)),
		torch.RandN([]int{H, Dout},
			at.TensorOptions().RequiresGrad(true)),
	}

	adam := optim.NewAdam(params, optim.AdamOptions(learning_rate))

	w1 := adam.parameters()[0]
	w2 := adam.parameters()[1]

	for i := 0; i < 500; i++ {
		y_pred := at.Sum(at.Clamp(at.MM(x, w1), 0), w2)
		loss := at.Sum(at.Pow(at.Sub(y_pred, y), 2))

		if i%100 == 0 {
			fmt.Println("loss = ", loss)
		}

		adam.ZeroGrad()
		loss.Backward()
		adam.Step()
	}
}
