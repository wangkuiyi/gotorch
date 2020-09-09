package tgz

import (
	"archive/tar"
	"bytes"
	"fmt"
	"image/color"
	"image/png"
	"path/filepath"
	"strings"

	"github.com/wangkuiyi/gotorch/vision"
)

// SynthesizeTarball generates a tarball file.
func SynthesizeTarball(dir string) (string, error) {
	fn := filepath.Join(dir, "input.tar.gz")

	w, e := CreateFile(fn)
	if e != nil {
		return "", e
	}

	if e := Synthesize(w); e != nil {
		return "", e
	}

	if e := w.Close(); e != nil {
		return "", e
	}

	return fn, e
}

// Synthesize generates a tar.gz file for testing.
func Synthesize(w *Writer) error {
	imgFns := []string{
		"mnist/",
		"mnist/training/",
		"mnist/training/0/",
		"mnist/training/0/first.png",
		"mnist/training/0/second.png",
		"mnist/training/1/",
		"mnist/training/1/first.png",
		"mnist/training/1/second.png",
		"mnist/testing/",
		"mnist/testing/0/",
		"mnist/testing/0/first.png",
	}

	var buf bytes.Buffer
	img := vision.SynthesizeImage(2, 2, color.RGBA{0, 0, 255, 255})
	if e := png.Encode(&buf, img); e != nil {
		return fmt.Errorf("Failed encoding PNG file: %v", e)
	}

	for _, fn := range imgFns {
		if strings.HasSuffix(fn, "/") { // A directory
			hdr := &tar.Header{
				Typeflag: tar.TypeDir,
				Name:     fn,
				Mode:     0600,
			}

			if e := w.WriteHeader(hdr); e != nil {
				return fmt.Errorf("Failed writing header: %v", e)
			}
		} else { // Regular file
			hdr := &tar.Header{
				Name: fn,
				Mode: 0600,
				Size: int64(buf.Len()),
			}

			if e := w.WriteHeader(hdr); e != nil {
				return fmt.Errorf("Failed writing header: %v", e)
			}
			if _, e := w.Write([]byte(buf.Bytes())); e != nil {
				return fmt.Errorf("Failed writing PNG encoding: %v", e)
			}
		}
	}

	return nil
}
