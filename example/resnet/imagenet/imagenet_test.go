package imagenet_test

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"image/color"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/example/resnet/imagenet"
)

func TestSynthesizer(t *testing.T) {
	var tgz bytes.Buffer
	s := imagenet.NewSynthesizer(&tgz)
	colors := []color.Color{
		color.RGBA{0, 0, 255, 255},
		color.RGBA{0, 255, 0, 255},
	}
	fns := []string{
		"/images/training/blue/01.jpeg",
		"/images/training/green/green.jpeg",
	}
	for i := 0; i < len(fns); i++ {
		s.AddImage(fns[i], 469, 387, colors[i])
	}
	s.Close()

	gr, e := gzip.NewReader(&tgz)
	assert.NoError(t, e)
	tr := tar.NewReader(gr)
	i := 0
	for {
		hdr, e := tr.Next()
		if e == io.EOF {
			break // End of archive
		}
		assert.NoError(t, e)
		assert.Equal(t, fns[i], hdr.Name)
		i++
	}
}
