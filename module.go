package gotorch

// #cgo CFLAGS: -I ${SRCDIR}/cgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch.h"
import "C"
import (
	"log"
	"math"
	"reflect"
)

// Module interface
type Module interface {
	Forward(x Tensor) Tensor
}

type linear struct {
	InFeatures  int
	OutFeatures int
	Weight      Tensor
	Bias        Tensor
}

// Linear creates a linear instance
func Linear(in int, out int, bias bool) Module {
	l := &linear{
		InFeatures:  in,
		OutFeatures: out,
	}
	l.Weight = RandN([]int{in, out}, true)
	if bias {
		l.Bias = RandN([]int{out, 1}, true)
	}
	return l
}

// Forward method
func (l *linear) Forward(x Tensor) Tensor {
	return MM(x, l.Weight)
}

type conv2d struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Dilation    int
	Groups      int
	HasBias     bool
	PaddingMode string
	Weight      Tensor
	Bias        Tensor
}

// Conv2d does conv2d computaion. torch.conv2d
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
func Conv2d(inChannels, outChannels, kernelSize, stride, padding, dilation,
	groups int, bias bool, paddingMode string) Module {
	c := &conv2d{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Dilation:    dilation,
		Groups:      groups,
		HasBias:     bias,
		PaddingMode: "zeros",
	}
	c.Weight = Empty([]int{outChannels, inChannels / groups, kernelSize,
		kernelSize}, true)
	KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if bias {
		c.Bias = Empty([]int{outChannels}, true)
		fanIn, _ := CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		Uniform(&c.Bias, -bound, bound)
	}
	return c
}

// Forward method
func (c *conv2d) Forward(x Tensor) Tensor {
	return FConv2d(x, c.Weight, c.Bias, []int{c.Stride, c.Stride},
		[]int{c.Padding, c.Padding}, []int{c.Dilation, c.Dilation}, c.Groups)
}

// GetNamedParameters returns parameters in the module recursively.
func GetNamedParameters(m Module) map[string]Tensor {
	r := make(map[string]Tensor)
	getNamedNonNilTensors(m, reflect.TypeOf(m).Elem().Name(), true, false, r)
	return r
}

func getNamedNonNilTensors(m Module, prefix string, param, buffer bool, r map[string]Tensor) {
	moduleType := reflect.TypeOf((*Module)(nil)).Elem()

	sv := reflect.ValueOf(m).Elem() // Elem gets what the pointer points to.
	for i := 0; i < sv.NumField(); i++ {
		f := sv.Type().Field(i)
		v := sv.Field(i)

		if f.Type.Implements(moduleType) {
			if !v.CanInterface() {
				log.Fatalf("GoTorch requires exporting Module field %s.%s",
					sv.Type().Name(), f.Name)
			}
			getNamedNonNilTensors(v.Interface().(Module),
				prefix+"."+f.Name, param, buffer, r)
		} else {
			recordNonNilTensor(f, v, prefix, r, param, buffer)
		}
	}
}

// If field f is a parameter or buffer field and the value v is not a nil
// tensor, insert v into map r with key is prefix+"."+f.Name.
func recordNonNilTensor(f reflect.StructField, v reflect.Value,
	prefix string, r map[string]Tensor, param, buffer bool) {

	tensorType := reflect.TypeOf((*Tensor)(nil)).Elem()
	if f.Type != tensorType {
		return // Either parameter or buffer is of type Tensor.
	}

	tag := f.Tag.Get("gotorch")
	if !buffer && tag == "buffer" {
		return // Don't wants a buffer but this field is one.
	}
	if !param && (tag == "param" || tag == "") {
		return // Don't wants a parameter but this field is one.
	}

	if !v.CanInterface() {
		log.Fatalf("GoTorch requires exporting Tensor field %s.%s",
			v.Type().Name(), f.Name)
	}

	fv := v.Interface().(Tensor)
	if fv.T == nil {
		return // Don't record nil Tensor
	}

	r[prefix+"."+f.Name] = fv
}

// GetParameters returns parameters
func GetParameters(m Module) []Tensor {
	result := make([]Tensor, 0)
	n := GetNamedParameters(m)
	for _, v := range n {
		result = append(result, v)
	}
	return result
}

// CloseModule closes the module
func CloseModule(m Module) {
	r := make(map[string]Tensor)
	getNamedNonNilTensors(m, reflect.TypeOf(m).Elem().Name(), true, true, r)
	for _, t := range r {
		t.Close()
	}
}
