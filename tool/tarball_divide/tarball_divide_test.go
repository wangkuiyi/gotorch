package main

import (
	"archive/tar"
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io/ioutil"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDivide(t *testing.T) {
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	if e != nil {
		t.Fatal(e)
	}

	fn := synthesizeTarball(t, d)

	l, e := ListTarGzFile(fn)
	assert.NoError(t, e)
	assert.Equal(t, 5, len(l))

	assert.NoError(t, divide(fn, d))

	l, e = ListTarGzFile(filepath.Join(d, "0.tar.gz"))
	assert.NoError(t, e)
	assert.Equal(t, 3, len(l))

	l, e = ListTarGzFile(filepath.Join(d, "1.tar.gz"))
	assert.NoError(t, e)
	assert.Equal(t, 2, len(l))
}

func synthesizeTarball(t *testing.T, dir string) string {
	fn := filepath.Join(dir, "input.tar.gz")

	w, e := CreatTarGzFile(fn)
	if e != nil {
		t.Fatalf("Cannot create writer: %v", e)
	}

	imgFns := []string{
		"mnist/training/0/first.png",
		"mnist/training/0/second.png",
		"mnist/training/1/first.png",
		"mnist/training/1/second.png",
		"mnist/testing/0/first.png",
	}

	for _, fn := range imgFns {
		var buf bytes.Buffer
		img := drawImage(image.Rect(0, 0, 2, 2), color.RGBA{0, 0, 255, 255})
		if e := png.Encode(&buf, img); e != nil {
			t.Fatalf("Failed encoding PNG file: %v", e)
		}

		hdr := &tar.Header{
			Name: fn,
			Mode: 0600,
			Size: int64(buf.Len()),
		}

		if e := w.WriteHeader(hdr); e != nil {
			t.Fatalf("Failed writing header: %v", e)
		}
		if _, e := w.Write([]byte(buf.Bytes())); e != nil {
			t.Fatalf("Failed writing PNG encoding: %v", e)
		}
	}

	if e := w.Close(); e != nil {
		t.Fatalf("Cannot close the synthesizer: %v", e)
	}

	return fn
}

func drawImage(size image.Rectangle, c color.Color) image.Image {
	m := image.NewRGBA(size)
	draw.Draw(m, m.Bounds(), &image.Uniform{c}, image.ZP, draw.Src)
	return m
}
