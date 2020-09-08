package tgz_test

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestTgzSynthesize(t *testing.T) {
	var buf bytes.Buffer

	w := tgz.NewWriter(&buf)
	assert.NoError(t, tgz.Synthesize(w))
	assert.NoError(t, w.Close())

	r, e := tgz.NewReader(&buf)
	assert.NoError(t, e)
	l, e := tgz.List(r)
	assert.NoError(t, e)
	for _, h := range l {
		t.Log(h.Name)
	}
}
