package tgz

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
)

// Reader includes a tar reader and its underlying readers.
type Reader struct {
	*tar.Reader
	g *gzip.Reader
	f *os.File
}

// OpenFile returns a Reader
func OpenFile(fn string) (*Reader, error) {
	f, e := os.Open(fn)
	if e != nil {
		return nil, e
	}

	r, e := NewReader(f)
	if e != nil {
		return nil, e
	}

	r.f = f
	return r, nil
}

// NewReader wraps an io.Reader into a Reader.
func NewReader(r io.Reader) (*Reader, error) {
	g, e := gzip.NewReader(r)
	if e != nil {
		return nil, e
	}

	return &Reader{
		Reader: tar.NewReader(g),
		g:      g,
	}, nil
}

// Close calls gzip.Reader.Close() and os.File.Close if the underlying storage
// is a file.
func (r *Reader) Close() error {
	if e := r.g.Close(); e != nil {
		return e
	}
	if r.f != nil {
		return r.f.Close()
	}
	return nil
}
