package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type plus struct {
	value int
}

func (t plus) Do(x int) int {
	return t.value + x
}

type div struct {
	value float32
}

func (t div) Do(a int) float32 {
	return float32(a) / t.value
}

type invalidTransform struct{}

func TestSequentialTransform(t *testing.T) {
	transforms := Sequential(&plus{10}, &div{2})
	// (2 + 10) / 2
	out := transforms.Do(2)
	assert.Equal(t, out, float32(6.0))
}

func TestSequentialTransformPanic(t *testing.T) {
	a := assert.New(t)
	a.Panics(func() {
		transforms := Sequential(&invalidTransform{})
		transforms.Do(10)
	}, "TestSequentialTransformPanic should panics")
}
