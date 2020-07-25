package main

// #cgo CFLAGS: -I ./cgotorch
// #cgo LDFLAGS: -Wl,-rpath ./cgotorch
// #cgo LDFLAGS: -L ./cgotorch -lcgotorch -L ./cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

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

func PrintTensor(a C.Tensor) {
	C.PrintTensor(a)
}

func Backward(a C.Tensor) {
	C.Backward(a)
}

func Grad(a C.Tensor) C.Tensor {
	return C.Grad(a)
}

func main() {
	a := RandN(3, 4, true)
	PrintTensor(a)

	b := RandN(4, 1, true)
	PrintTensor(b)

	c := MM(a, b)
	PrintTensor(c)

	d := Sum(c)
	PrintTensor(d)

	Backward(d)

	PrintTensor(Grad(a))
	PrintTensor(Grad(b))
}
