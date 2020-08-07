package gotorch_test

import (
	torch "github.com/wangkuiyi/gotorch"
)

func ExampleTensor() {
	t := torch.RandN(10, 100, false)
	t.Close()
	t.Close()
	// Output:
}

func ExampleTranspose2d() {
	input := torch.RandNByShape([]int{1, 1, 1}, false)
	weight := torch.RandNByShape([]int{1, 3, 3}, false)
	var bias torch.Tensor
	stride := []int{1}
	padding := []int{0}
	outputPadding := []int{0}
	groups := 1
	dilation := []int{1}
	out := torch.ConvTranspose2d(input, weight, bias,
		stride, padding, outputPadding, groups, dilation)
	out.Print()
}
