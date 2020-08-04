package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	t := C.RandN(C.int(rows), C.int(cols), C.int(rg))
	setTensorFinalizer(&t)
	return Tensor{&t}
}
