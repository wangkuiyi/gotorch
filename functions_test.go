package gotorch_test

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

func ExampleMMException() {
	defer func() {
		torch.GC()
		if r := recover(); r != nil {
			fmt.Println("Recovered:", r.(string)[:45])
		}
		torch.FinishGC()
	}()

	x := torch.RandN([]int64{10, 20}, true)
	y := torch.RandN([]int64{200, 300}, true)
	z := torch.MM(x, y)
	_ = z
	// Output: Recovered: size mismatch, m1: [10 x 20], m2: [200 x 300]
}

func ExampleMM() {
	x := torch.RandN([]int64{10, 20}, true)
	y := torch.RandN([]int64{20, 30}, true)
	z := torch.MM(x, y)
	_ = z
	// Output:
}

func ExampleRelu() {
	x := torch.RandN([]int64{10, 20}, true)
	r := torch.Relu(x)
	r = torch.LeakyRelu(x, 0.01)
	_ = r
	// TODO(shendiaomo): more tests when other function wrapper available
	// Output:
}
