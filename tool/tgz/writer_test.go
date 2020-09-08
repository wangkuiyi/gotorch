package tgz_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestTgzCreateFileNoPermission(t *testing.T) {
	w, e := tgz.CreateFile("/")
	assert.Nil(t, w)
	assert.Error(t, e)
}

func TestTgzNewWriterFromNil(t *testing.T) {
	w := tgz.NewWriter(nil)
	assert.Nil(t, w)
}
