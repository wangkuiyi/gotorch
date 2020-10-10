package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
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

// Ones return a tensor filled with 1
func Ones(shape []int64, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(
		unsafe.Pointer(C.Ones((*C.int64_t)(unsafe.Pointer(&shape[0])),
			C.int64_t(len(shape)), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Eye returns a tensor with 1s on diagonal and 0s elsewhere
func Eye(n, m int64, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Eye(C.int64_t(n), C.int64_t(m), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Full returns a tensor with all elements being given `value`
func Full(shape []int64, value float32, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Full((*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int64_t(len(shape)), C.float(value), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Arange returns a 1-D tensor in range [begin, end) with
// common difference step beginning from begin
func Arange(begin, end, step float32, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Arange(C.float(begin),
		C.float(end), C.float(step), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Linspace returns a 1-D Tensor in range [begin, end] with steps points
func Linspace(begin, end float32, steps int64, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Linspace(C.float(begin),
		C.float(end), C.int64_t(steps), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Logspace returns a 1-D Tensor of steps points
// logarithmically spaced with base base between
// pow(base, begin) and pow(base, end)
func Logspace(begin, end float32, steps int64,
	base float64, requiresGrad bool) Tensor {
	rg := 0
	if requiresGrad {
		rg = 1
	}
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Logspace(C.float(begin),
		C.float(end), C.int64_t(steps),
		C.double(base), C.int64_t(rg), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}
