package transforms

import (
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToTensor(t *testing.T) {
	// image to Tensor
	m := generateRandImage(image.Rect(0, 0, 4, 4))
	trans := ToTensor()
	out := trans.Run(m)
	assert.Equal(t, out.Shape(), []int64{4, 4, 3})
	// int to Tensor
	out = trans.Run(10)
	assert.Equal(t, out.Shape(), []int64{1})
}
