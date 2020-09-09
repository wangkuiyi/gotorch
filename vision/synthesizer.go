package vision

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
)

// SynthesizeImage creates an image of specified size and uniform color.  To
// generate a blue image of generate ImageNet size (469x387), we can call
//
//    blue := color.RGBA{0, 0, 255, 255}
//    m := SynthesizeImage(469, 387, blue)
func SynthesizeImage(w, h int, c color.Color) image.Image {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(m, m.Bounds(), &image.Uniform{c}, image.ZP, draw.Src)
	return m
}

// Synthesizer synthesize a tar.gz file of images.
type Synthesizer struct {
	gzw *gzip.Writer
	*tar.Writer
}

// NewSynthesizer creates a synthesizer.
func NewSynthesizer(output io.Writer) *Synthesizer {
	gzw := gzip.NewWriter(output)
	return &Synthesizer{
		gzw:    gzw,
		Writer: tar.NewWriter(gzw),
	}
}

// Close a synthesizer.
func (s *Synthesizer) Close() {
	s.Writer.Close()
	s.gzw.Close()
	s.Writer, s.gzw = nil, nil
}

// AddImage synthesize an image with given filename, size, and color.
func (s *Synthesizer) AddImage(fn string, w, h int, c color.Color) error {
	var im bytes.Buffer
	e := jpeg.Encode(&im, SynthesizeImage(w, h, c), nil)
	if e != nil {
		return e
	}

	e = s.WriteHeader(&tar.Header{
		Typeflag: tar.TypeReg,
		Name:     fn,
		Mode:     0600,
		Size:     int64(im.Len()),
	})
	if e != nil {
		return e
	}

	_, e = s.Write(im.Bytes())
	return e
}
