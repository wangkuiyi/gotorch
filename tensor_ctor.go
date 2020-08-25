package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"

import (
	"unsafe"
)

// RandN returns a tensor filled with standard normal distribution, torch.randn
func RandN(shape []int64, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.RandN((*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int64_t(len(shape)), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Rand torch.rand
func Rand(shape []int64, requireGrad bool) Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Rand((*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int64_t(len(shape)), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Empty returns a tensor filled with random number, torch.empty
func Empty(shape []int64, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(
		unsafe.Pointer(C.Empty((*C.int64_t)(unsafe.Pointer(&shape[0])),
			C.int64_t(len(shape)), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}
