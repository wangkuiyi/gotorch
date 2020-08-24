package gotorch

import (
	"log"
	"reflect"
	"unsafe"

	"github.com/wangkuiyi/gotorch/variadic"
)

const (
	// Invalid Dtype
	Invalid = -1
	// Byte Dtype 0
	Byte int8 = iota
	// Char Dtype 1
	Char
	// Short Dtype 2
	Short
	// Int Dtype 3
	Int
	// Long Dtype 4
	Long
	// Half Dtype 5
	Half
	// Float Dtype 6
	Float
	// Double Dtype 7
	Double
	// ComplexHalf Dtype 8
	ComplexHalf
	// ComplexFloat Dtype 9
	ComplexFloat
	// ComplexDouble Dtype 10
	ComplexDouble
	// Bool Dtype 11
	Bool
	// QInt8 Dtype 12
	QInt8
	// QUInt8 Dtype 13
	QUInt8
	// QInt32 Dtype 14
	QInt32
	// BFloat16 Dtype 15
	BFloat16
)

// NewTensor creates a tensor from a Go slice.  We use variadic parameters of
// type map[string]interface{} to mimic named variadic parameters.
func NewTensor(data interface{}, options ...map[string]interface{}) Tensor {
	t := reflect.TypeOf(data)
	if t.Kind() != reflect.Slice {
		log.Panicf("NewTensor requires a slice; got a %v", t.Kind())
	}

	// TODO(wangkuiyi): Add Tensor.{SetRequiresGrad,RequiresGrad}.
	// requireGrad := variadic.Get(options, "requires_grad").(bool)

	shape, kind := sliceShapeAndElemKind(data)
	dtype := tensorElemDType(options, kind)

	sliceAddr := unsafe.Pointer(&data)
	dataAddr := *(*uintptr)(sliceAddr)
	return FromBlob(unsafe.Pointer(dataAddr), dtype, shape)
}

func sliceShapeAndElemKind(data interface{}) ([]int64, reflect.Kind) {
	var r []int64
	v := reflect.ValueOf(data)
	for {
		k := v.Type().Kind()
		if k != reflect.Slice {
			return r, k
		}
		r = append(r, int64(v.Len()))
		v = v.Index(0)
	}
	return nil, reflect.Invalid
}

func tensorElemDType(options []map[string]interface{}, k reflect.Kind) int8 {
	// If the user specified the DType explicitly, it overrides the one
	// derived from Go type.
	//
	// TODO(wangkuiyi): Check the size of the specified Dtype matches that
	// of the Go element type.
	if dtype, ok := variadic.Lookup(options, "dtype"); ok {
		return dtype.(int8)
	}

	dtype, ok := goTypeToTorch[k]
	if !ok {
		return Invalid
	}
	return dtype

}

var (
	goTypeToTorch = map[reflect.Kind]int8{
		reflect.Bool:      Bool,
		reflect.Int:       Int,
		reflect.Int8:      Byte,
		reflect.Int16:     Half,
		reflect.Uint16:    Half,
		reflect.Float32:   Float,
		reflect.Float64:   Double,
		reflect.Complex64: ComplexDouble,
	}
)
