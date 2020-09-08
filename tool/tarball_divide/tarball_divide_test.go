package main

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestDivide(t *testing.T) {
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	if e != nil {
		t.Fatal(e)
	}

	fn, e := tgz.SynthesizeTarball(d)
	assert.NoError(t, e)

	l, e := tgz.ListFile(fn)
	assert.NoError(t, e)
	assert.Equal(t, 5, len(l))

	assert.NoError(t, divide(fn, d))

	l, e = tgz.ListFile(filepath.Join(d, "0.tar.gz"))
	assert.NoError(t, e)
	assert.Equal(t, 3, len(l))

	l, e = tgz.ListFile(filepath.Join(d, "1.tar.gz"))
	assert.NoError(t, e)
	assert.Equal(t, 2, len(l))
}
