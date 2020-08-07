package gotorch_test

import (
	torch "github.com/wangkuiyi/gotorch"
)

func ExampleTensor() {
	t := torch.RandN([]int{10, 100}, false)
	t.Close()
	t.Close()
	// Output:
}
