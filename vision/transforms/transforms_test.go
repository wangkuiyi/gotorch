package transforms

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type plus struct {
	value int
}

func (t plus) Run(x int) int {
	return t.value + x
}

type div struct {
	value float32
}

func (t div) Run(a int) float32 {
	return float32(a) / t.value
}

type invalidTransform struct{}

type divAndMod struct {
	value int
}

func (t divAndMod) Run(x int) (int, int) {
	return x / t.value, x % t.value
}

func TestEmptyTransform(t *testing.T) {
	transforms := Compose()
	out := transforms.Run(10)
	assert.Equal(t, out, 10)
}

func TestSequentialTransform(t *testing.T) {
	transforms := Compose(&plus{10}, &div{2})
	// (2 + 10) / 2
	out := transforms.Run(2)
	assert.Equal(t, out, float32(6.0))
}

func TestSequentialTransformPanic(t *testing.T) {
	a := assert.New(t)
	a.Panics(func() {
		transforms := Compose(&invalidTransform{})
		transforms.Run(10)
	}, "TestSequentialTransformPanic should panics")
}

func TestTransformReturnMoreThanOneValuePanic(t *testing.T) {
	assert.Panics(t, func() {
		transforms := Compose(&divAndMod{10})
		transforms.Run(21)
	})
}
