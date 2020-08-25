package datasets_test

import (
	"bytes"
	"image/color"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/vision/datasets"
)

func TestImgNetLoader(t *testing.T) {
	var tgz1, tgz2 bytes.Buffer
	w := io.MultiWriter(&tgz1, &tgz2)
	generateColorData(w)
	vocab, err := datasets.BuildLabelVocabulary(&tgz1)
	assert.NoError(t, err)

	loader, err := datasets.ImageNet(&tgz2, vocab, 2)
	assert.NoError(t, err)

	for loader.Scan() {
		data, label := loader.Minibatch().Data, loader.Minibatch().Target
		assert.Equal(t, []int64{2, 469, 387, 3}, data.Shape())
		assert.Equal(t, []int64{2, 1}, label.Shape())
	}
}

func TestToTensor(t *testing.T) {
	{
		// image to Tensor
		blue := color.RGBA{0, 0, 255, 255}
		m := datasets.SynthesizeImage(4, 4, blue)
		out, err := datasets.ToTensor(m)
		assert.NoError(t, err)
		assert.Equal(t, out.Shape(), []int64{4, 4, 3})
	}
	{
		// int to Tensor
		out, err := datasets.ToTensor(10)
		assert.NoError(t, err)
		assert.Equal(t, out.Shape(), []int64{1})
	}
}
