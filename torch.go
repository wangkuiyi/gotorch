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
	Opt C.Optimizer
}

// RandN returns a tensor filled with random number
func RandN(rows, cols int, requireGrad bool) *Tensor {
	rg := 0
	if requireGrad {
		rg = 1
	}
	t := &Tensor{C.RandN(C.int(rows), C.int(cols), C.int(rg))}
	runtime.SetFinalizer(t, func(f *Tensor) {
		f.Close()
	})
	return t
}

// NewSGDOpt creates a SGD Optimizer
func NewSGDOpt(lr, momentum, dampening, weightDecay float64, nesterov bool) *Optimizer {
	nt := 0
	if nesterov {
		nt = 1
	}
	o := &Optimizer{
		C.SGD(C.double(lr), C.double(momentum), C.double(dampening),
			C.double(weightDecay), C.int(nt))}

	runtime.SetFinalizer(o, func(f *Optimizer) {
		f.Close()
	})
	return o
}

// AddParameters adds parameters
func (opt *Optimizer) AddParameters(tensors []*Tensor) {
	CT := []unsafe.Pointer{}
	for _, t := range tensors {
		CT = append(CT, unsafe.Pointer(t.T))
	}
	p := (*reflect.SliceHeader)(unsafe.Pointer(&CT)).Data
	C.AddParameters(opt.Opt, (*C.Tensor)(unsafe.Pointer(p)), C.int(len(CT)))
}

// ZeroGrad reset gradients to zero
func (opt *Optimizer) ZeroGrad() {
	C.ZeroGrad(opt.Opt)
}

// Step updates parameters
func (opt *Optimizer) Step() {
	C.Step(opt.Opt)
}

// Close the optimizer
func (opt *Optimizer) Close() {
	C.Optimizer_Close(opt.Opt)
}
