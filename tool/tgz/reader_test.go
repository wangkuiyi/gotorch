package tgz_test

import (
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
	_, e := tgz.NewReader(a)
	assert.Error(t, e) // the input is not valid a tarball encoding.
}
