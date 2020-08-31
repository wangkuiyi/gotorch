package tgz

import (
	"archive/tar"
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"path/filepath"
	"testing"
)

// SynthesizeTarball generates a tar.gz file for testing.
func SynthesizeTarball(t *testing.T, dir string) string {
	fn := filepath.Join(dir, "input.tar.gz")

	w, e := CreateFile(fn)
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
