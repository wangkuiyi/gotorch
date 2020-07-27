package main

// #cgo CFLAGS: -I ./cgotorch
// #cgo LDFLAGS: -L ./cgotorch -Wl,-rpath ./cgotorch -lcgotorch
// #cgo LDFLAGS: -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

type Tensor struct {
	T C.Tensor
}

func RandN(rows, cols int, require_grad bool) Tensor {
	rg := 0
	if require_grad {
		rg = 1
	}
	return Tensor{C.RandN(C.int(rows), C.int(cols), C.int(rg))}
}

func MM(a, b Tensor) Tensor {
	return Tensor{C.MM(a.T, b.T)}
}

func Sum(a Tensor) Tensor {
	return Tensor{C.Sum(a.T)}
}

func (a Tensor) Print() {
	C.Tensor_Print(a.T)
}

func (a Tensor) Backward() {
	C.Tensor_Backward(a.T)
}

func (a Tensor) Grad() Tensor {
	return Tensor{C.Tensor_Grad(a.T)}
}

func main() {
	a := RandN(3, 4, true)
	a.Print()

	b := RandN(4, 1, true)
	b.Print()

	c := MM(a, b)
	c.Print()

	d := Sum(c)
	d.Print()

	d.Backward()

	a.Grad().Print()
	b.Grad().Print()
}
