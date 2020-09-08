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
	assert.Equal(t, 5, len(l))
	fns := []string{
		"mnist/training/0/first.png",
		"mnist/training/0/second.png",
		"mnist/training/1/first.png",
		"mnist/training/1/second.png",
		"mnist/testing/0/first.png"}
	for i, h := range l {
		assert.Equal(t, fns[i], h.Name)
	}
}

func TestTgzSynthesizeWithoutPermission(t *testing.T) {
	fn, e := tgz.SynthesizeTarball("/somewhere_not_there")
	assert.Error(t, e)
	assert.Equal(t, "", fn)
}
