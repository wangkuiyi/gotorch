package tgz_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestOpenFileNotExist(t *testing.T) {
	r, e := tgz.OpenFile("not-there")
	assert.Nil(t, r)
	assert.Error(t, e)
}

func TestNewReaderInvalidTarball(t *testing.T) {
	a := strings.NewReader("    ")
	r, e := tgz.NewReader(a)
	assert.Error(t, e) // the input is not valid a tarball encoding.
	assert.Nil(t, r)
}

func TestNewReaderValidTarball(t *testing.T) {
	var buf bytes.Buffer
	w := tgz.NewWriter(&buf)
	assert.NoError(t, tgz.Synthesize(w))

	_, e := tgz.NewReader(&buf)
	assert.NoError(t, e) // the input is not valid a tarball encoding.
}
