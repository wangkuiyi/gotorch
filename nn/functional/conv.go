package functional

// #cgo CFLAGS: -I ${SRCDIR}/../../cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch -Wl,-rpath ${SRCDIR}/../../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "../../cgotorch/cgotorch.h"
import "C"

import (
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
)

// Conv2d does 2d-convolution
func Conv2d(input, weight, bias torch.Tensor,
	stride, padding, dilation []int, groups int) torch.Tensor {
	var cbias, t C.Tensor
	if bias.T != nil {
		cbias = C.Tensor(bias.T)
	}
	torch.MustNil(unsafe.Pointer(C.Conv2d(
		C.Tensor(input.T),
		C.Tensor(weight.T),
		cbias,
		(*C.int64_t)(unsafe.Pointer(&stride[0])), C.int64_t(len(stride)),
		(*C.int64_t)(unsafe.Pointer(&padding[0])), C.int64_t(len(padding)),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])), C.int64_t(len(dilation)),
		C.int64_t(groups),
		&t)))
	torch.SetTensorFinalizer(&t)
	return Tensor{unsafe.Pointer(t)}
}

// ConvTranspose2d does 2d-fractionally-strided convolution
func ConvTranspose2d(
	input, weight, bias torch.Tensor,
	stride, padding, outputPadding []int,
	groups int, dilation []int) torch.Tensor {

	var cbias, t C.Tensor
	if bias.T != nil {
		cbias = *bias.T
	}

	torch.MustNil(C.ConvTranspose2d(
		*input.T,
		*weight.T,
		cbias,
		(*C.int64_t)(unsafe.Pointer(&stride[0])),
		C.int64_t(len(stride)),
		(*C.int64_t)(unsafe.Pointer(&padding[0])),
		C.int64_t(len(padding)),
		(*C.int64_t)(unsafe.Pointer(&outputPadding[0])),
		C.int64_t(len(outputPadding)),
		C.int64_t(groups),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])),
		C.int64_t(len(dilation)),
		&t))
	torch.SetTensorFinalizer(&t)
	return torch.Tensor{&t}
}
