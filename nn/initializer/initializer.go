package initializer

// #cgo CFLAGS: -I ${SRCDIR}/../..
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch -Wl,-rpath ${SRCDIR}/../../cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/../../cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/../../cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	"log"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
)

// ManualSeed set the random seed
func ManualSeed(seed int64) {
	C.ManualSeed(C.int64_t(seed))
}

// Zeros initialization, torch.nn.init.zeros_
func Zeros(a *torch.Tensor) {
	if a == nil || a.T == nil {
		log.Panicf("Normal: input tensor is nil")
	}
	torch.MustNil(unsafe.Pointer(C.Zeros_((*C.Tensor)(a.T))))
}

// Ones initialization, torch.nn.init.ones_
func Ones(a *torch.Tensor) {
	if a == nil || a.T == nil {
		log.Panicf("Normal: input tensor is nil")
	}
	torch.MustNil(unsafe.Pointer(C.Ones_((*C.Tensor)(a.T))))
}

// Uniform initialization, torch.nn.init.uniform_
func Uniform(a *torch.Tensor, low, high float64) {
	if a == nil || a.T == nil {
		log.Panicf("Normal: input tensor is nil")
	}
	torch.MustNil(unsafe.Pointer(C.Uniform_((*C.Tensor)(a.T), C.double(low), C.double(high))))
}

// Normal initialization, torch.nn.init.normal_
func Normal(a *torch.Tensor, mean, std float64) {
	if a == nil || a.T == nil {
		log.Panicf("Normal: input tensor is nil")
	}
	torch.MustNil(unsafe.Pointer(C.Normal_((*C.Tensor)(a.T), C.double(mean), C.double(std))))
}

// KaimingUniform initialization, torch.nn.init.kaiming_uniform_
func KaimingUniform(input *torch.Tensor, a float64, fanMode string,
	nonLinearity string) {
	if input == nil || input.T == nil {
		log.Panicf("Normal: input tensor is nil")
	}
	torch.MustNil(unsafe.Pointer(C.KaimingUniform_(C.double(a), C.CString(fanMode),
		C.CString(nonLinearity), (*C.Tensor)(input.T))))
}

// CalculateFanInAndFanOut torch.nn.init._calculate_fan_in_and_fan_out
func CalculateFanInAndFanOut(input torch.Tensor) (int64, int64) {
	var fanIn, fanOut int64
	torch.MustNil(unsafe.Pointer(C.CalculateFanInAndFanOut(
		C.Tensor(*input.T),
		(*C.int64_t)(unsafe.Pointer(&fanIn)),
		(*C.int64_t)(unsafe.Pointer(&fanOut)))))
	return fanIn, fanOut
}
