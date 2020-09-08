package main

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestMergeFiles(t *testing.T) {
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	if e != nil {
		t.Fatal(e)
	}

	fn1, e := tgz.SynthesizeTarball(d)
	assert.NoError(t, e)
	fn2, e := tgz.SynthesizeTarball(d)
	assert.NoError(t, e)

	out := filepath.Join(d, "merged.tar.gz")
	mergeFiles([]string{fn1, fn2}, out)

	l, e := tgz.ListFile(out)
	assert.NoError(t, e)
	for i := 0; i < len(l); i = i + 2 {
		assert.Equal(t, l[i].Name, l[i+1].Name)
	}
}
