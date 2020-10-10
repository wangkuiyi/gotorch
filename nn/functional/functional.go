package functional

// #cgo CFLAGS: -I ${SRCDIR}/../..
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch -Wl,-rpath ${SRCDIR}/../../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	"runtime"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
)

// BatchNorm does batch nomalization for `input`
func BatchNorm(input, runningMean, runningVar, weight, bias torch.Tensor,
	training bool, momentum, eps float64) torch.Tensor {
	var cTraining C.int8_t
	if training {
		cTraining = 1
	}
	var cweight, cbias, cmean, cvar, t C.Tensor
	if runningMean.T != nil {
		cmean = C.Tensor(*runningMean.T)
	}
	if runningVar.T != nil {
		cvar = C.Tensor(*runningVar.T)
	}
	if weight.T != nil {
		cweight = C.Tensor(*weight.T)
	}
	if bias.T != nil {
		cbias = C.Tensor(*bias.T)
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
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// Conv2d does 2d-convolution
func Conv2d(input, weight, bias torch.Tensor,
	stride, padding, dilation []int64, groups int64) torch.Tensor {
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
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// ConvTranspose2d does 2d-fractionally-strided convolution
func ConvTranspose2d(
	input, weight, bias torch.Tensor,
	stride, padding, outputPadding []int64,
	groups int64, dilation []int64) torch.Tensor {

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
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// LogSoftmax torch.nn.functional.log_softmax
func LogSoftmax(input torch.Tensor, dim int64) torch.Tensor {
	// Clone _get_softmax_dim()
	if dim < 0 {
		if d := input.Dim(); d == 0 || d == 1 || d == 3 {
			dim = 0
		} else {
			dim = 1
		}
	}
	return input.LogSoftmax(dim)
}

// NllLoss torch.nn.functional.nll_loss
func NllLoss(input, target, weight torch.Tensor, ignoreIndex int64,
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
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// BinaryCrossEntropy torch.nn.functional.binary_cross_entropy
func BinaryCrossEntropy(input, target, weight torch.Tensor,
	reduction string) torch.Tensor {
	var cweight, t C.Tensor
	if weight.T != nil {
		cweight = C.Tensor(*weight.T)
	}
	torch.MustNil(unsafe.Pointer(C.BinaryCrossEntropy(
		C.Tensor(*input.T),
		C.Tensor(*target.T),
		cweight,
		C.CString(reduction),
		&t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// CrossEntropy torch.nn.functional.cross_entropy
func CrossEntropy(input, target, weight torch.Tensor, ignoreIndex int64,
	reduction string) torch.Tensor {
	var cweight, t C.Tensor
	if weight.T != nil {
		cweight = C.Tensor(*weight.T)
	}
	torch.MustNil(unsafe.Pointer(C.CrossEntropy(
		C.Tensor(*input.T),
		C.Tensor(*target.T),
		cweight,
		C.int64_t(ignoreIndex),
		C.CString(reduction),
		&t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// Relu torch.nn.functional.relu
func Relu(input torch.Tensor, inplace bool) torch.Tensor {
	var t C.Tensor
	var cInplace C.int8_t
	if inplace {
		cInplace = 1
	}
	torch.MustNil(unsafe.Pointer(C.FRelu(
		C.Tensor(*input.T), C.int8_t(cInplace), &t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// LeakyRelu torch.nn.functional.leaky_relu
func LeakyRelu(input torch.Tensor, negativeSlope float64, inplace bool) torch.Tensor {
	var t C.Tensor
	var cInplace C.int8_t
	if inplace {
		cInplace = 1
	}
	torch.MustNil(unsafe.Pointer(C.FLeakyRelu(C.Tensor(*input.T),
		C.double(negativeSlope), C.int8_t(cInplace), &t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// Linear ports torch.nn.functional.linear
func Linear(input, weight, bias torch.Tensor) torch.Tensor {
	var t C.Tensor
	var cBias C.Tensor
	if bias.T != nil {
		cBias = C.Tensor(*bias.T)
	}
	torch.MustNil(unsafe.Pointer(C.Linear(
		C.Tensor(*input.T),
		C.Tensor(*weight.T), cBias, &t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// MaxPool2d torch.nn.functional.max_pool2d
func MaxPool2d(input torch.Tensor, kernelSize, stride, padding,
	dilation []int64, ceilMode bool) torch.Tensor {
	var cMode C.int8_t
	if ceilMode {
		cMode = 1
	}
	var t C.Tensor
	torch.MustNil(unsafe.Pointer(C.MaxPool2d(
		C.Tensor(*input.T),
		(*C.int64_t)(unsafe.Pointer(&kernelSize[0])), C.int64_t(len(kernelSize)),
		(*C.int64_t)(unsafe.Pointer(&stride[0])), C.int64_t(len(stride)),
		(*C.int64_t)(unsafe.Pointer(&padding[0])), C.int64_t(len(padding)),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])), C.int64_t(len(dilation)),
		cMode,
		&t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}

// AdaptiveAvgPool2d torch.nn.functional.adaptive_avg_pool2d
func AdaptiveAvgPool2d(input torch.Tensor, outputSize []int64) torch.Tensor {
	var t C.Tensor
	torch.MustNil(unsafe.Pointer(C.AdaptiveAvgPool2d(
		C.Tensor(*input.T),
		(*C.int64_t)(unsafe.Pointer(&outputSize[0])), C.int64_t(len(outputSize)),
		&t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}
