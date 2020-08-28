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
}
