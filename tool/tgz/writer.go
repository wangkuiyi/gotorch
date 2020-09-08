package tgz

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
)

// Writer defines a writer to a .tar.gz file.
type Writer struct {
	*tar.Writer
	g *gzip.Writer
	f *os.File
}

// CreateFile creates a writer.
func CreateFile(fn string) (*Writer, error) {
	f, e := os.Create(fn)
	if e != nil {
		return nil, e
	}

	w := NewWriter(f)
	w.f = f
	return w, nil
}

// NewWriter wraps an io.Writer into a Writer
func NewWriter(r io.Writer) *Writer {
	if r == nil {
		return nil
	}
	g := gzip.NewWriter(r)
	return &Writer{
		Writer: tar.NewWriter(g),
		g:      g,
		f:      nil}
}

// Close calls the tar.Writer.Close(), gzip.Writer.Close(), and if the
// underlying storage is a file, os.File.Close().
func (w *Writer) Close() error {
	if e := w.Writer.Close(); e != nil {
		return e
	}
	if e := w.g.Close(); e != nil {
		return e
	}
	if w.f != nil {
		return w.f.Close()
	}
	return nil
}
