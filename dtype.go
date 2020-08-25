package gotorch

import (
	"log"
	"reflect"
	"unsafe"

	"github.com/wangkuiyi/gotorch/variadic"
)

const (
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
	// Invalid Dtype
	Invalid = -1
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
	if dtype == Invalid {
		log.Panicf("Unrecognized element kind %v", kind)
	}
	f := flattenSlice(data, kind)
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&f))
	return FromBlob(unsafe.Pointer(hdr.Data), dtype, shape)
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
	// The user specified DType, if there is any, overrides the one derived
	// from Go reflection.
	//
	// TODO(wangkuiyi): Check the size of the specified Dtype matches that
	// of the Go element type.
	if dtype, ok := variadic.Lookup(options, "dtype"); ok {
		return dtype.(int8)
	}
	dtype, ok := goTypeToTorch[k]
	if !ok {
		dtype = Invalid
	}
	return dtype
}

var (
	goTypeToTorch = map[reflect.Kind]int8{
		reflect.Bool:    Bool,
		reflect.Int:     Int,
		reflect.Int8:    Byte,
		reflect.Uint16:  Half,
		reflect.Float32: Float,
		reflect.Float64: Double,
	}
)

// https://medium.com/@the1mills/flattening-arrays-slices-with-golang-c796905debbe
// provides a way to flatten any recursive slices into []interface{}.  However,
// we cannot reuse this solution here, because libtorch wants
// []float32/float64/..., instead of []interface{}.  Without type template as
// that in C++, we have to write the following Go code repeatedly.
func flattenSlice(slc interface{}, kind reflect.Kind) unsafe.Pointer {
	switch kind {
	case reflect.Bool:
		f := flattenSliceBool(nil, reflect.ValueOf(slc))
		return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
	case reflect.Int:
		f := flattenSliceInt(nil, reflect.ValueOf(slc))
		return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
	case reflect.Int8:
		f := flattenSliceInt8(nil, reflect.ValueOf(slc))
		return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
	case reflect.Uint16:
		f := flattenSliceUint16(nil, reflect.ValueOf(slc))
		return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
	case reflect.Float32:
		f := flattenSliceFloat32(nil, reflect.ValueOf(slc))
		return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
	case reflect.Float64:
		f := flattenSliceFloat64(nil, reflect.ValueOf(slc))
		return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
	}
	return nil
}

func flattenSliceFloat32(args []float32, v reflect.Value) []float32 {
	if v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			args = flattenSliceFloat32(args, v.Index(i))
		}
	} else {
		args = append(args, float32(v.Float()))
	}
	return args
}

func flattenSliceFloat64(args []float64, v reflect.Value) []float64 {
	if v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			args = flattenSliceFloat64(args, v.Index(i))
		}
	} else {
		args = append(args, v.Float())
	}
	return args
}

func flattenSliceUint16(args []uint16, v reflect.Value) []uint16 {
	if v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			args = flattenSliceUint16(args, v.Index(i))
		}
	} else {
		args = append(args, uint16(v.Uint()))
	}
	return args
}

func flattenSliceBool(args []bool, v reflect.Value) []bool {
	if v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			args = flattenSliceBool(args, v.Index(i))
		}
	} else {
		args = append(args, v.Bool())
	}
	return args
}

func flattenSliceInt(args []int, v reflect.Value) []int {
	if v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			args = flattenSliceInt(args, v.Index(i))
		}
	} else {
		args = append(args, int(v.Int()))
	}
	return args
}

func flattenSliceInt8(args []int8, v reflect.Value) []int8 {
	if v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			args = flattenSliceInt8(args, v.Index(i))
		}
	} else {
		args = append(args, int8(v.Int()))
	}
	return args
}
