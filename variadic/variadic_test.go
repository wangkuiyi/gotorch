package variadic_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/variadic"
)

func TestHas(t *testing.T) {
	assert.True(t, variadic.Has(opts, "dtype"))
	assert.True(t, variadic.Has(opts, "requires_grad"))
	assert.False(t, variadic.Has(opts, "RequiresGrad"))
	assert.False(t, variadic.Has(nil, "anything"))
}

func TestGet(t *testing.T) {
	assert.Equal(t, 111, variadic.Get(opts, "dtype"))
	assert.Equal(t, false, variadic.Get(opts, "requires_grad"))
	assert.Equal(t, nil, variadic.Get(opts, "RequiresGrad"))
	assert.Equal(t, nil, variadic.Get(nil, "anything"))
	assert.Equal(t, nil, variadic.Get(opts, "anything"))
	assert.Equal(t, 33, variadic.Get(opts, "anything", 33).(int))
}

func TestLookup(t *testing.T) {
	v, ok := variadic.Lookup(opts, "dtype")
	assert.Equal(t, 111, v)
	assert.True(t, ok)

	v, ok = variadic.Lookup(opts, "requires_grad")
	assert.Equal(t, false, v)
	assert.True(t, ok)

	v, ok = variadic.Lookup(opts, "RequiresGrad")
	assert.Equal(t, nil, v)
	assert.False(t, ok)

	v, ok = variadic.Lookup(nil, "anything")
	assert.Equal(t, nil, v)
	assert.False(t, ok)
}

var (
	opts = []map[string]interface{}{{
		"dtype":         111,
		"requires_grad": false,
	}}
)
