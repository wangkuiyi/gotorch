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
	a := assert.New(t)
	var tgz bytes.Buffer
	synthesizeImages(&tgz)

	vocab, e := datasets.BuildLabelVocabulary(bytes.NewReader(tgz.Bytes()))
	a.NoError(e)
	a.Equal(3, len(vocab))

	trans := transforms.Compose(transforms.RandomCrop(224, 224), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor())
	loader, e := datasets.ImageNet(bytes.NewReader(tgz.Bytes()), vocab, trans, 2)
	a.NoError(e)
	{
		// the first iteration
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{2, 3, 224, 224}, data.Shape())
		a.Equal([]int64{2}, label.Shape())
		a.NoError(loader.Err())
	}
	{
		// the second iteration returns the last minibatch which contains
		// one sample only.
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{1, 3, 224, 224}, data.Shape())
		a.Equal([]int64{1}, label.Shape())
		a.NoError(loader.Err())
	}
	// no more data
	a.False(loader.Scan())
}

func TestBuildLabelVocabularyFail(t *testing.T) {
	_, e := datasets.BuildLabelVocabulary(strings.NewReader("some string"))
	assert.Error(t, e)
}
