package tgz_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func TestCreateFileNoPermission(t *testing.T) {
	w, e := tgz.CreateFile("/")
	assert.Nil(t, w)
	assert.Error(t, e)
}
