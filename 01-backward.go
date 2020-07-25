package main

// #cgo CFLAGS: -I ./cgotorch
// #cgo LDFLAGS: -Wl,-rpath ./cgotorch
// #cgo LDFLAGS: -L ./cgotorch -lcgotorch -L ./cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import "fmt"

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

func (a Tensor) String() string {
	s := C.Tensor_String(a.T)
	r := C.GoString(s)
	C.FreeString(s)
	return r
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
	fmt.Println(a)

	b := RandN(4, 1, true)
	fmt.Println(b)

	c := MM(a, b)
	fmt.Println(c)

	d := Sum(c)
	fmt.Println(d)

	d.Backward()

	fmt.Println(a.Grad())
	fmt.Println(b.Grad())
}
