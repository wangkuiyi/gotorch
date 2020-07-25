package main

// #cgo CFLAGS: -I ./libtorch/include
// #cgo CFLAGS: -I ./libtorch/include/torch/csrc/api/include
// #cgo CFLAGS: -I ./cgotorch
// #cgo LDFLAGS: -L ./cgotorch -L ./cgotorch/libtorch/lib
// #cgo LDFLAGS: -lcgotorch -lc10 -ltorch -ltorch_cpu
// #include "ctorch.h"
import "C"

import (
	"fmt"
)

func RandN(rows, cols int, require_grad bool) C.Tensor {
	rg := 0
	if require_grad {
		rg = 1
	}
	return C.RandN(C.int(rows), C.int(cols), C.int(rg))
}

func MM(a, b C.Tensor) C.Tensor {
	return C.MM(a, b)
}

func Sum(a C.Tensor) C.Tensor {
	return C.Sum(a)
}

func main() {
	a := RandN(3, 4, true)
	fmt.Println("a = ", a)

	b := RandN(4, 1, true)
	fmt.Println("b = ", b)

	c := MM(a, b)
	fmt.Println("c = ", c)

	d := Sum(c)
	fmt.Println("d = ", d)

	// d.Backward()

	// fmt.Println("a.grad = ", a.Grad())
	// fmt.Println("b.grad = ", b.Grad())
}
