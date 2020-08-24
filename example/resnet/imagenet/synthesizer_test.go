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

func generateColorData(w io.Writer) []string {
	s := imagenet.NewSynthesizer(w)
	defer s.Close()
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
	return fns
}

func TestSynthesizer(t *testing.T) {
	var tgz bytes.Buffer
	fns := generateColorData(&tgz)

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

func TestDataloader(t *testing.T) {
	var tgz bytes.Buffer
	generateColorData(&tgz)
	loader, err := imagenet.NewDataLoader(&tgz, 4)
	assert.NoError(t, err)

	for loader.Scan() {
		loader.Batch()
	}
}
