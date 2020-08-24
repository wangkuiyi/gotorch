package imagenet_test

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/example/resnet/imagenet"
)

func TestDataloader(t *testing.T) {
	var tgz bytes.Buffer
	generateColorData(&tgz)
	loader, err := imagenet.NewDataLoader(&tgz, 4)
	assert.NoError(t, err)

	for loader.Scan() {
		loader.Batch()
	}
}
