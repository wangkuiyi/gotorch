package gotorch

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/x448/float16"
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

func f16(x float32) uint16 {
	return float16.Fromfloat32(x).Bits()
}

func TestNewTensor(t *testing.T) {
	assert.Equal(t, " 1  0\n 0  1\n[ CPUBoolType{2,2} ]",
		NewTensor([][]bool{{true, false}, {false, true}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUByteType{2,2} ]",
		NewTensor([][]uint8{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUByteType{2,2} ]",
		NewTensor([][]byte{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUCharType{2,2} ]",
		NewTensor([][]int8{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUShortType{2,2} ]",
		NewTensor([][]int16{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUIntType{2,2} ]",
		NewTensor([][]int32{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPULongType{2,2} ]",
		NewTensor([][]int64{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUHalfType{2,2} ]",
		NewTensor([][]uint16{{f16(1), f16(0)}, {f16(0), f16(1)}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUFloatType{2,2} ]",
		NewTensor([][]float32{{1, 0}, {0, 1}}).String())

	assert.Equal(t, " 1  0\n 0  1\n[ CPUDoubleType{2,2} ]",
		NewTensor([][]float64{{1, 0}, {0, 1}}).String())
}

func TestNewTensorUnsupportGoTypes(t *testing.T) {
	assert.Panics(t, func() { NewTensor([]uint32{1, 0}) })
	assert.Panics(t, func() { NewTensor([]uint64{1, 0}) })
	assert.Panics(t, func() { NewTensor([]uintptr{1, 0}) })
	assert.Panics(t, func() { NewTensor([]int{1, 0}) })

	// TODO(wangkuiyi): Need to support complex64
	assert.Panics(t, func() { NewTensor([]complex64{1 + 1i, -1 - 1i}) })
	assert.Panics(t, func() { NewTensor([]complex128{1 + 1i, -1 - 1i}) })
}
