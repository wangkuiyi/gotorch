package torch

// #cgo CFLAGS: -I ${SRCDIR}/../cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch -Wl,-rpath ${SRCDIR}/../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"unsafe"
	"github.com/wangkuiyi/gotorch/aten"
)
// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) aten.Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	return aten.NewTensor(unsafe.Pointer(C.RandN(C.int(rows),  C.int(cols), C.int(rg))))
}
