package imagenet_test

import (
	"bytes"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/example/resnet/imagenet"
)

func TestDataloader(t *testing.T) {
	var tgz1, tgz2 bytes.Buffer
	w := io.MultiWriter(&tgz1, &tgz2)
	generateColorData(w)
	vob, err := imagenet.BuildLabelVocabulary(&tgz1)
	assert.NoError(t, err)

	loader, err := imagenet.NewDataLoader(&tgz2, vob, 4)
	assert.NoError(t, err)

	for loader.Scan() {
		loader.Minibatch()
	}
}
