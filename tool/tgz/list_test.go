package tgz_test

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestTgzListFile(t *testing.T) {
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	if e != nil {
		t.Fatal(e)
	}

	fn, e := tgz.SynthesizeTarball(d)
	assert.NoError(t, e)

	l, e := tgz.ListFile(fn)
	assert.NoError(t, e)
	assert.Equal(t, 5, len(l))
}

func TestTgzListNotExistingFile(t *testing.T) {
	l, e := tgz.ListFile("somefile_not_there")
	assert.Error(t, e)
	assert.Nil(t, l)
}
