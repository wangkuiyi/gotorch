package imagenet_test

import (
	"bytes"
	"image/color"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/example/resnet/imagenet"
)

func TestDataloader(t *testing.T) {
	var tgz1, tgz2 bytes.Buffer
	w := io.MultiWriter(&tgz1, &tgz2)
	generateColorData(w)
	vocab, err := imagenet.BuildLabelVocabulary(&tgz1)
	assert.NoError(t, err)

	loader, err := imagenet.NewDataLoader(&tgz2, vocab, 2)
	assert.NoError(t, err)

	for loader.Scan() {
		data, label := loader.Minibatch()
		assert.Equal(t, []int64{2, 469, 387, 3}, data.Shape())
		assert.Equal(t, []int64{2, 1}, label.Shape())
	}
}

func TestToTensor(t *testing.T) {
	{
		// image to Tensor
		blue := color.RGBA{0, 0, 255, 255}
		m := imagenet.SynthesizeImage(4, 4, blue)
		out, err := imagenet.ToTensor(m)
		assert.NoError(t, err)
		assert.Equal(t, out.Shape(), []int64{4, 4, 3})
	}
	{
		// int to Tensor
		out, err := imagenet.ToTensor(10)
		assert.NoError(t, err)
		assert.Equal(t, out.Shape(), []int64{1})
	}
}
