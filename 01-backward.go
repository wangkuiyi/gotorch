package main

// #cgo CFLAGS: -I ./cgotorch
// #cgo LDFLAGS: -L ./cgotorch -Wl,-rpath ./cgotorch -lcgotorch
// #cgo LDFLAGS: -L ./cgotorch/libtorch/lib -Wl,-rpath ./cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import "fmt"

// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T C.Tensor
}

// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	return Tensor{C.RandN(C.int(rows), C.int(cols), C.int(rg))}
}

// MM multiplies each element of the input two tensors
func MM(a, b Tensor) Tensor {
	return Tensor{C.MM(a.T, b.T)}
}

// Sum returns the sum of all elements in the input tensor
func Sum(a Tensor) Tensor {
	return Tensor{C.Sum(a.T)}
}

// String returns the Tensor as a string
func (a Tensor) String() string {
	s := C.Tensor_String(a.T)
	r := C.GoString(s)
	C.FreeString(s)
	return r
}

// Print the tensor
func (a Tensor) Print() {
	C.Tensor_Print(a.T)
}

// Backward computes the gradient of current tensor
func (a Tensor) Backward() {
	C.Tensor_Backward(a.T)
}

// Grad returns a reference of the gradient
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
