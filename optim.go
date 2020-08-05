package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"reflect"
	"runtime"
	"unsafe"
)

// Optimizer struct
type Optimizer struct {
	Opt *C.Optimizer
}

// SGD creates a SGD Optimizer
func SGD(lr, momentum, dampening, weightDecay float64, nesterov bool) Optimizer {
	nt := 0
	if nesterov {
		nt = 1
	}
	sgd := C.SGD(C.double(lr), C.double(momentum), C.double(dampening),
		C.double(weightDecay), C.int(nt))
	runtime.SetFinalizer(&sgd, func(p *C.Optimizer) { C.Optimizer_Close(*p) })
	return Optimizer{&sgd}
}

// Adam creates an Adam Optimizer
func Adam(lr, beta1, beta2, weightDecay float64) Optimizer {
	adam := C.Adam(C.double(lr), C.double(beta1), C.double(beta2), C.double(weightDecay))
	runtime.SetFinalizer(&adam, func(p *C.Optimizer) { C.Optimizer_Close(*p) })
	return Optimizer{&adam}
}

// AddParameters adds parameters
func (opt Optimizer) AddParameters(tensors []Tensor) {
	CT := []unsafe.Pointer{}
	for _, t := range tensors {
		CT = append(CT, unsafe.Pointer(*t.T))
	}
	p := (*reflect.SliceHeader)(unsafe.Pointer(&CT)).Data
	C.Optimizer_AddParameters(*opt.Opt, (*C.Tensor)(unsafe.Pointer(p)), C.int(len(CT)))
}

// ZeroGrad reset gradients to zero
func (opt Optimizer) ZeroGrad() {
	C.ZeroGrad(*opt.Opt)
}

// Step updates parameters
func (opt Optimizer) Step() {
	C.Step(*opt.Opt)
}

// Close the optimizer
func (opt Optimizer) Close() {
	C.Optimizer_Close(*opt.Opt)
}
