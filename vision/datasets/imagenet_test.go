package datasets_test

import (
	"bytes"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func TestImgNetLoader(t *testing.T) {
	var tgz1, tgz2 bytes.Buffer
	w := io.MultiWriter(&tgz1, &tgz2)
	generateColorData(w)
	vocab, err := datasets.BuildLabelVocabulary(&tgz1)
	assert.NoError(t, err)
	trans := transforms.Compose(transforms.RandomCrop(224, 224), transforms.RandomFlip(), transforms.ToTensor())
	loader, err := datasets.ImageNet(&tgz2, vocab, trans, 2)
	assert.NoError(t, err)
	for loader.Scan() {
		data, label := loader.Minibatch()
		assert.Equal(t, []int64{2, 3, 224, 224}, data.Shape())
		assert.Equal(t, []int64{2}, label.Shape())
	}
}
