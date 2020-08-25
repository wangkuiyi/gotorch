package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func TestRandN(t *testing.T) {
	a := torch.RandN([]int64{10, 100}, false)
	assert.Equal(t, []int64{10, 100}, a.Shape())
	assert.NotPanics(t, func() {
		a.Close()
		a.Close()
	})
}

func TestRand(t *testing.T) {
	a := torch.Rand([]int64{50, 100}, false)
	assert.Equal(t, []int64{50, 100}, a.Shape())
	assert.NotPanics(t, func() {
		a.Close()
		a.Close()
	})
}

func TestEmpty(t *testing.T) {
	a := torch.Empty([]int64{50, 10}, false)
	assert.Equal(t, []int64{50, 10}, a.Shape())
	assert.NotPanics(t, func() {
		a.Close()
		a.Close()
	})
}
