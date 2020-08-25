package gotorch

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSliceShapeAndElemKind(t *testing.T) {
	{
		data := [][]float64{{1, 2}, {3, 4}, {5, 6}}
		shape, kind := sliceShapeAndElemKind(data)
		assert.Equal(t, []int64{3, 2}, shape)
		assert.Equal(t, reflect.Float64, kind)
	}
	{
		data := [][]float32{{1, 2}}
		shape, kind := sliceShapeAndElemKind(data)
		assert.Equal(t, []int64{1, 2}, shape)
		assert.Equal(t, reflect.Float32, kind)
	}
	{
		data := int32(1)
		shape, kind := sliceShapeAndElemKind(data)
		assert.Equal(t, 0, len(shape))
		assert.Equal(t, reflect.Int32, kind)
	}
}

func TestTensorElemDType(t *testing.T) {
	{
		data := []uint16{1, 2, 3}
		shape, kind := sliceShapeAndElemKind(data)
		assert.Equal(t, []int64{3}, shape)

		dtype := tensorElemDType(
			[]map[string]interface{}{{"dtype": Bool}}, kind)
		assert.Equal(t, Bool, dtype) // Half overrides int16

		dtype = tensorElemDType(nil, kind)
		assert.Equal(t, Half, dtype) // Deriving Half from int16
	}
	{
		data := []float32{1, 2, 3}
		shape, kind := sliceShapeAndElemKind(data)
		assert.Equal(t, []int64{3}, shape)

		dtype := tensorElemDType(nil, kind)
		assert.Equal(t, Float, dtype)
	}
}

func TestNewTensor(t *testing.T) {
	{
		a := NewTensor([][]float32{{1.0, 1.1, 1.2}, {2, 3, 4}})
		assert.Equal(t, []int64{2, 3}, a.Shape())
		assert.Equal(t, Float, a.Dtype())
	}
	{
		a := NewTensor([][]uint16{{1, 2}, {3, 4}})
		assert.Equal(t, []int64{2, 2}, a.Shape())
		assert.Equal(t, Half, a.Dtype())
	}
	{
		a := NewTensor([]int8{1, 2, 3})
		assert.Equal(t, []int64{3}, a.Shape())
		assert.Equal(t, Byte, a.Dtype())
	}
	{
		assert.Panics(t, func() {
			// int16 cannot be converted into PyTorch type
			NewTensor([][]int16{{1, 2}, {3, 4}})
		})
	}
}

func TestFlattenSlice(t *testing.T) {
	{
		d := [][]float32{{1, 2, 3}, {4, 5, 6}}
		f := flattenSliceFloat32(nil, reflect.ValueOf(d))
		assert.Equal(t, 6, len(f))
		assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, f)
	}
	{
		d := [][]float64{{1, 2}, {3, 4}, {5, 6}}
		f := flattenSliceFloat64(nil, reflect.ValueOf(d))
		assert.Equal(t, 6, len(f))
		assert.Equal(t, []float64{1, 2, 3, 4, 5, 6}, f)
	}
	{
		d := [][]bool{{true, false}, {false, true}}
		f := flattenSliceBool(nil, reflect.ValueOf(d))
		assert.Equal(t, 4, len(f))
		assert.Equal(t, []bool{true, false, false, true}, f)
	}
	{
		d := []int{1, 2}
		f := flattenSliceInt(nil, reflect.ValueOf(d))
		assert.Equal(t, 2, len(f))
		assert.Equal(t, []int{1, 2}, f)
	}
	{
		d := [][]uint16{{1, 2}, {3, 4}}
		f := flattenSliceUint16(nil, reflect.ValueOf(d))
		assert.Equal(t, 4, len(f))
		assert.Equal(t, []uint16{1, 2, 3, 4}, f)
	}
	{
		d := [][]int8{{1, 2, 3}, {4, 5, 6}}
		f := flattenSliceInt8(nil, reflect.ValueOf(d))
		assert.Equal(t, 6, len(f))
		assert.Equal(t, []int8{1, 2, 3, 4, 5, 6}, f)
	}
}
