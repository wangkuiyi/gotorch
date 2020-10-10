package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"
import (
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
		C.double(weightDecay), C.int64_t(nt))
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
	CT := []C.Tensor{}
	for _, t := range tensors {
		CT = append(CT, C.Tensor(*t.T))
	}
	p := (*C.Tensor)(unsafe.Pointer(&CT[0]))
	C.Optimizer_AddParameters(*opt.Opt, p, C.int64_t(len(CT)))
}

// ZeroGrad reset gradients to zero
func (opt Optimizer) ZeroGrad() {
	C.Optimizer_ZeroGrad(*opt.Opt)
}

// Step updates parameters
func (opt Optimizer) Step() {
	C.Optimizer_Step(*opt.Opt)
}

// SetLR sets learning rate
func (opt Optimizer) SetLR(lr float64) {
	C.Optimizer_SetLR(*opt.Opt, C.double(lr))
}

// Close the optimizer
func (opt Optimizer) Close() {
	C.Optimizer_Close(*opt.Opt)
}
