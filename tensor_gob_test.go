package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch"
)

func TestTensorGobEncode(t *testing.T) {
	a := gotorch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	b, e := a.GobEncode()
	assert.NoError(t, e)
	// The ground-truth length comes from the C++ program example/pickle.
	assert.Equal(t, 747, len(b))
	// TODO(wangkuiyi): verify the content of []byte.
	// TODO(wangkuiyi): test encoding an empty tensor.
}

func TestTensorGobDecode(t *testing.T) {
	x := gotorch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	b, e := x.GobEncode()
	assert.NoError(t, e)
	// The ground-truth length comes from the C++ program example/pickle.
	assert.Equal(t, 747, len(b))

	y, e := gotorch.GobDecodeTensor(b)
	assert.NoError(t, e)
	assert.Equal(t, " 1  0  0\n 0  1  0\n 0  0  1\n[ CPUFloatType{3,3} ]", y.String())
}
