package gotorch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import "C"

import (
	"log"
	"reflect"
	"strings"
	"unsafe"

	"github.com/wangkuiyi/gotorch/variadic"
)

// Add torch.add
func Add(a, other Tensor, alpha float32) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Add(C.Tensor(*a.T), C.Tensor(*other.T),
		C.float(alpha), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Add torch.add
func (a *Tensor) Add(other Tensor, alpha float32) Tensor {
	return Add(*a, other, alpha)
}

// AddI adds in-place
func (a *Tensor) AddI(other Tensor, alpha float32) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Add_(
		C.Tensor(*a.T),
		C.Tensor(*other.T),
		C.float(alpha),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Sub torch.sub
func Sub(a, other Tensor, alpha float32) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Sub(C.Tensor(*a.T), C.Tensor(*other.T),
		C.float(alpha), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Sub torch.sub
func (a *Tensor) Sub(other Tensor, alpha float32) Tensor {
	return Sub(*a, other, alpha)
}

// SubI subs in-place
func (a *Tensor) SubI(other Tensor, alpha float32) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Sub_(
		C.Tensor(*a.T),
		C.Tensor(*other.T),
		C.float(alpha),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Mul torch.mul
func Mul(a, other Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Mul(C.Tensor(*a.T), C.Tensor(*other.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Mul torch.Mul
func (a *Tensor) Mul(other Tensor) Tensor {
	return Mul(*a, other)
}

// MulI multiplies in-place
func (a *Tensor) MulI(other Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Mul_(
		C.Tensor(*a.T),
		C.Tensor(*other.T),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Div torch.div
func Div(a, other Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Div(C.Tensor(*a.T), C.Tensor(*other.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Div torch.Div
func (a *Tensor) Div(other Tensor) Tensor {
	return Div(*a, other)
}

// DivI run divides in-place
func (a *Tensor) DivI(other Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Div_(
		C.Tensor(*a.T),
		C.Tensor(*other.T),
		&t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Permute transpose the tensor dims.
func (a *Tensor) Permute(dims []int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Permute(C.Tensor(*a.T), (*C.int64_t)(&dims[0]), C.int64_t(len(dims)), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Eq wraps torch.eq, which does element-wise comparison between two tensors and returns
// a tensor of the same size as the operands.
func Eq(a, other Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Eq(C.Tensor(*a.T), C.Tensor(*other.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Eq torch.eq
func (a Tensor) Eq(other Tensor) Tensor {
	return Eq(a, other)
}

// Equal compares two tensors by their content.
func Equal(a, b Tensor) bool {
	var r int64
	MustNil(unsafe.Pointer(C.Equal(C.Tensor(*a.T), C.Tensor(*b.T), (*C.int64_t)(&r))))
	return r != 0
}

// AllClose returns true if the float tensor are all close.
func AllClose(a, b Tensor) bool {
	var r int64
	MustNil(unsafe.Pointer(C.AllClose(C.Tensor(*a.T), C.Tensor(*b.T), (*C.int64_t)(&r))))
	return r != 0
}

// ExpandAs torch.expand_as
func ExpandAs(a, other Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.ExpandAs(C.Tensor(*a.T), C.Tensor(*other.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// ExpandAs torch.expand_as
func (a Tensor) ExpandAs(other Tensor) Tensor {
	return ExpandAs(a, other)
}

// Flatten torch.flatten
func Flatten(a Tensor, startDim, endDim int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Flatten(C.Tensor(*a.T), C.int64_t(startDim), C.int64_t(endDim), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// IndexSelect torch.index_select
func IndexSelect(a Tensor, dim int64, index Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.IndexSelect(C.Tensor(*a.T), C.int64_t(dim), C.Tensor(*index.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// IndexSelect torch.index_select
func (a Tensor) IndexSelect(dim int64, index Tensor) Tensor {
	return IndexSelect(a, dim, index)
}

// Item returns 0-dim tensor's value as an interface
// users should do type assertion and get the value like:
// v, ok := a.Item().(float64)
// Currently not support unsigned Tensor.
func (a Tensor) Item() interface{} {
	dtype := a.Dtype()
	switch dtype {
	case Byte, Bool, Char, Short, Int, Long:
		var v int64
		MustNil(unsafe.Pointer(C.ItemInt64(C.Tensor(*a.T), (*C.int64_t)(&v))))
		switch dtype {
		case Byte:
			return byte(v)
		case Bool:
			return bool(v != 0)
		case Char:
			return int8(v)
		case Short:
			return int16(v)
		case Int:
			return int32(v)
		case Long:
			return v
		}
	case Half, Float, Double:
		var v float64
		MustNil(unsafe.Pointer(C.ItemFloat64(C.Tensor(*a.T), (*C.double)(&v))))
		switch dtype {
		case Half, Float:
			return float32(v)
		case Double:
			return v
		}
	}
	log.Panicf("DType %d not supported now.", a.Dtype())
	return nil
}

// LeakyRelu returns leaky relu of the tensor according to negativeSlope
func LeakyRelu(t Tensor, negativeSlope float64) Tensor {
	return t.LeakyRelu(negativeSlope)
}

// LeakyRelu returns leaky relu of the tensor according to negativeSlope
func (a *Tensor) LeakyRelu(negativeSlope float64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.LeakyRelu(C.Tensor(*a.T), C.double(negativeSlope), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// LogSoftmax returns log softmax of the input tensor
func LogSoftmax(t Tensor, dim int64) Tensor {
	return t.LogSoftmax(dim)
}

// LogSoftmax returns log softmax of the current tensor
func (a Tensor) LogSoftmax(dim int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.LogSoftmax(C.Tensor(*a.T), C.int64_t(dim), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Mean returns mean of the current tensor
func Mean(t Tensor) Tensor {
	return t.Mean()
}

// Mean torch.mean
func (a Tensor) Mean() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Mean(C.Tensor(*a.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// MM multiplies each element of the input two tensors
func MM(a, b Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.MM(C.Tensor(*a.T), C.Tensor(*b.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Relu returns relu of the tensor
func (a *Tensor) Relu() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Relu(C.Tensor(*a.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Relu returns relu of the tensor
func Relu(t Tensor) Tensor {
	return t.Relu()
}

// Sigmoid returns sigmoid of the current tensor
func Sigmoid(t Tensor) Tensor {
	return t.Sigmoid()
}

// Sigmoid returns sigmoid of the current tensor
func (a Tensor) Sigmoid() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Sigmoid(C.Tensor(*a.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Stack concatenates sequence of tensors along a new dimension
func Stack(tensors []Tensor, dim int64) Tensor {
	CT := []C.Tensor{}
	for _, t := range tensors {
		CT = append(CT, C.Tensor(*t.T))
	}
	p := (*C.Tensor)(unsafe.Pointer(&CT[0]))
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Stack(p, C.int64_t(len(CT)), C.int64_t(dim), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Squeeze torch.squeeze
func Squeeze(t Tensor, dim ...int64) Tensor {
	return t.Squeeze(dim...)
}

// Squeeze tensor.squeeze
func (a Tensor) Squeeze(dim ...int64) Tensor {
	var t C.Tensor
	switch len(dim) {
	case 0:
		MustNil(unsafe.Pointer(C.Squeeze(C.Tensor(*a.T), &t)))
		SetTensorFinalizer((*unsafe.Pointer)(&t))
		return Tensor{(*unsafe.Pointer)(&t)}
	case 1:
		MustNil(unsafe.Pointer(C.SqueezeWithDim(C.Tensor(*a.T), C.int64_t(dim[0]), &t)))
		SetTensorFinalizer((*unsafe.Pointer)(&t))
		return Tensor{(*unsafe.Pointer)(&t)}
	default:
		panic("Squeeze only accepts 0-1 dim as input")
	}
}

// Sum is torch.sum
func Sum(a Tensor, opt ...map[string]interface{}) Tensor {
	if variadic.Has(opt, "dim") {
		dim := variadic.Get(opt, "dim").(int)
		keepDim := variadic.Get(opt, "keepDim", false).(bool)

		k := 0
		if keepDim {
			k = 1
		}
		var t C.Tensor
		MustNil(unsafe.Pointer(C.SumByDim(C.Tensor(*a.T), C.int64_t(dim), C.int8_t(k), &t)))
		SetTensorFinalizer((*unsafe.Pointer)(&t))
		return Tensor{(*unsafe.Pointer)(&t)}
	}

	var t C.Tensor
	MustNil(unsafe.Pointer(C.Sum(C.Tensor(*a.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Sum is Tensor.sum
func (a Tensor) Sum(opt ...map[string]interface{}) Tensor {
	return Sum(a, opt...)
}

// Tanh returns tanh of the current tensor
func Tanh(t Tensor) Tensor {
	return t.Tanh()
}

// Tanh returns tanh of the current tensor
func (a Tensor) Tanh() Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Tanh(C.Tensor(*a.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// TopK torch.topk
func TopK(a Tensor, k, dim int64, largest, sorted bool) (Tensor, Tensor) {
	var values, indices C.Tensor
	l := 0
	if largest {
		l = 1
	}
	s := 0
	if sorted {
		s = 1
	}
	MustNil(unsafe.Pointer(C.TopK(C.Tensor(*a.T), C.int64_t(k), C.int64_t(dim),
		C.int8_t(l), C.int8_t(s), &values, &indices)))
	SetTensorFinalizer((*unsafe.Pointer)(&values))
	SetTensorFinalizer((*unsafe.Pointer)(&indices))
	return Tensor{(*unsafe.Pointer)(&values)}, Tensor{(*unsafe.Pointer)(&indices)}
}

// Transpose torch.transpose
func Transpose(a Tensor, dim0, dim1 int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.Transpose(C.Tensor(*a.T), C.int64_t(dim0), C.int64_t(dim1), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// Transpose torch.transpose
func (a Tensor) Transpose(dim0, dim1 int64) Tensor {
	return Transpose(a, dim0, dim1)
}

// View returns a new Tensor with the same data but of a different shape
func View(a Tensor, shape ...int64) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.View(C.Tensor(*a.T), &t, (*C.int64_t)(unsafe.Pointer(&shape[0])), C.int64_t(len(shape)))))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}

// View returns a new Tensor with the same data but of a different shape
func (a Tensor) View(shape ...int64) Tensor {
	return View(a, shape...)
}

// Argmin mimics torch.argmin
func (a Tensor) Argmin(opts ...interface{}) Tensor {
	return a.argMinMax(true, opts...)
}

// Argmax mimics torch.argmax
func (a Tensor) Argmax(opts ...interface{}) Tensor {
	return a.argMinMax(false, opts...)
}

func (a Tensor) argMinMax(argmin bool, opts ...interface{}) Tensor {
	var (
		dimOpt  int64
		dim     *int64
		keepdim int8
	)

	if len(opts) > 0 {
		// The first optional parameter must be dim integer.
		if !strings.HasPrefix(reflect.TypeOf(opts[0]).Kind().String(), "int") {
			log.Panicf("Tensor.Argmin(dim) requires dim in int{64|32|16|}")
		}
		dimOpt = reflect.ValueOf(opts[0]).Int()
		dim = &dimOpt
	}

	if len(opts) > 1 {
		// The second optional parametr must be keepdim bool.
		if reflect.TypeOf(opts[1]).Kind() != reflect.Bool {
			log.Panicf("Tensor.Argmin(dim) requires dim in int64")
		}
		if opts[1].(bool) {
			keepdim = 1
		}
	}

	var t C.Tensor
	if argmin {
		MustNil(unsafe.Pointer(C.Argmin(C.Tensor(*a.T), (*C.int64_t)(dim), C.int8_t(keepdim), &t)))
	} else {
		MustNil(unsafe.Pointer(C.Argmax(C.Tensor(*a.T), (*C.int64_t)(dim), C.int8_t(keepdim), &t)))
	}
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}
