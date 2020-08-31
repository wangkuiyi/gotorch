package tgz_test

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestOpenFileNotExist(t *testing.T) {
	r, e := tgz.OpenFile("not-there")
	assert.Nil(t, r)
	assert.Error(t, e)
}

func TestCreateFileNoPermission(t *testing.T) {
	w, e := tgz.CreateFile("/")
	assert.Nil(t, w)
	assert.Error(t, e)
}

func TestListFile(t *testing.T) {
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	if e != nil {
		t.Fatal(e)
	}

	fn := tgz.SynthesizeTarball(t, d)

	l, e := tgz.ListFile(fn)
	assert.NoError(t, e)
	assert.Equal(t, 5, len(l))
}
