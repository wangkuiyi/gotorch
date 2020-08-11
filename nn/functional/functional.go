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

// BatchNorm does batch nomalization for `input`
func BatchNorm(input, weight, bias, runningMean, runningVar torch.Tensor,
	training bool, momentum, eps float64) torch.Tensor {
	var cTraining C.int8_t
	if training {
		cTraining = 1
	}
	var cweight, cbias, cmean, cvar, t C.Tensor
	if weight.T != nil {
		cweight = C.Tensor(*weight.T)
	}
	if bias.T != nil {
		cbias = C.Tensor(*bias.T)
	}
	if runningMean.T != nil {
		cmean = C.Tensor(*runningMean.T)
	}
	if runningVar.T != nil {
		cvar = C.Tensor(*runningVar.T)
	}
	torch.MustNil(
		unsafe.Pointer(C.BatchNorm(
			C.Tensor(*input.T),
			cweight,
			cbias,
			cmean,
			cvar,
			cTraining,
			C.double(momentum),
			C.double(eps),
			&t)))
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// Conv2d does 2d-convolution
func Conv2d(input, weight, bias torch.Tensor,
	stride, padding, dilation []int, groups int) torch.Tensor {
	var cbias, t C.Tensor
	if bias.T != nil {
		cbias = C.Tensor(*bias.T)
	}
	torch.MustNil(unsafe.Pointer(C.Conv2d(
		C.Tensor(*input.T),
		C.Tensor(*weight.T),
		cbias,
		(*C.int64_t)(unsafe.Pointer(&stride[0])), C.int64_t(len(stride)),
		(*C.int64_t)(unsafe.Pointer(&padding[0])), C.int64_t(len(padding)),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])), C.int64_t(len(dilation)),
		C.int64_t(groups),
		&t)))
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// ConvTranspose2d does 2d-fractionally-strided convolution
func ConvTranspose2d(
	input, weight, bias torch.Tensor,
	stride, padding, outputPadding []int,
	groups int, dilation []int) torch.Tensor {

	var cbias, t C.Tensor
	if bias.T != nil {
		cbias = C.Tensor(*bias.T)
	}

	torch.MustNil(unsafe.Pointer(C.ConvTranspose2d(
		C.Tensor(*input.T),
		C.Tensor(*weight.T),
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
		&t)))
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// NllLoss torch.nn.functional.nll_loss
func NllLoss(input, target, weight torch.Tensor, ignoreIndex int,
	reduction string) torch.Tensor {
	var cweight, t C.Tensor
	if weight.T != nil {
		cweight = C.Tensor(*weight.T)
	}
	torch.MustNil(unsafe.Pointer(C.NllLoss(
		C.Tensor(*input.T),
		C.Tensor(*target.T),
		cweight,
		C.int64_t(ignoreIndex),
		C.CString(reduction),
		&t)))
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}
