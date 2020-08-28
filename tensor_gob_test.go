package gotorch_test

import (
	"bytes"
	"crypto/md5"
	"encoding/gob"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch"
)

func TestTensorGobEncode(t *testing.T) {
	a := gotorch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	b, e := a.GobEncode()
	assert.NoError(t, e)
	// The ground-truth length and MD5 checksum come from the C++ program
	// example/pickle.
	assert.Equal(t, 747, len(b))
	assert.Equal(t, fmt.Sprintf("%x", md5.Sum(b)), "dd65752601bf4d4ca19ae903baf96799")

	a = gotorch.Tensor{nil}
	_, e = a.GobEncode()
	assert.Error(t, e)
}

func TestTensorGobDecode(t *testing.T) {
	x := gotorch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})

	var buf bytes.Buffer
	assert.NoError(t, gob.NewEncoder(&buf).Encode(x))

	var y gotorch.Tensor
	assert.NoError(t, gob.NewDecoder(&buf).Decode(&y))
	assert.Equal(t, " 1  0  0\n 0  1  0\n 0  0  1\n[ CPUFloatType{3,3} ]", y.String())
}
