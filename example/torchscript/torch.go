package main

import (
	"fmt"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/jit"
)

func testModel() {
	model := jit.LoadJITModule("model.pt")
	fmt.Println("Model loaded successfully!")
	inputTensor := torch.RandN([]int64{1, 1, 240, 320}, false)
	fmt.Println("Input tensor created successfully!")
	res := model.Forward(inputTensor)
	fmt.Printf("res is tuple: %v\n", res.IsTuple())
	tuple := res.ToTuple()
	for i, t := range tuple {
		fmt.Printf("tuple[%d] is tensor: %v\n", i, t.IsTensor())
		t.ToTensor().Print()
	}
}

func main() {
	testModel()
}
