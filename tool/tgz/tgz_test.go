package tgz_test

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestDivide(t *testing.T) {
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	if e != nil {
		t.Fatal(e)
	}

	fn := tgz.SynthesizeTarball(t, d)

	l, e := tgz.ListFile(fn)
	assert.NoError(t, e)
	assert.Equal(t, 5, len(l))
}

func TestListFile(t *testing.T) {
	l, e := tgz.ListFile("/tmp/mnist_png.tar.gz")
	assert.NoError(t, e)
	t.Log(len(l))
}
