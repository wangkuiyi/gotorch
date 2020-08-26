package datasets_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func TestImgNetLoader(t *testing.T) {
	var tgz bytes.Buffer
	synthesizeImages(&tgz)

	vocab, e := datasets.BuildLabelVocabulary(bytes.NewReader(tgz.Bytes()))
	assert.NoError(t, e)
	assert.Equal(t, 2, len(vocab))

	trans := transforms.Compose(transforms.RandomCrop(224, 224), transforms.RandomFlip(), transforms.ToTensor())
	loader, e := datasets.ImageNet(bytes.NewReader(tgz.Bytes()), vocab, trans, 2)
	assert.NoError(t, e)
	for loader.Scan() {
		data, label := loader.Minibatch()
		assert.Equal(t, []int64{2, 3, 224, 224}, data.Shape())
		assert.Equal(t, []int64{2}, label.Shape())
	}
	assert.NoError(t, loader.Err())
}

func TestBuildLabelVocabularyFail(t *testing.T) {
	_, e := datasets.BuildLabelVocabulary(strings.NewReader("some string"))
	assert.Error(t, e)
}
