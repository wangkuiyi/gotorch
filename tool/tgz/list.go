package tgz

import (
	"archive/tar"
	"io"
)

// ListFile list contents in a .tar.gz file.
func ListFile(fn string) ([]*tar.Header, error) {
	r, e := OpenFile(fn)
	if e != nil {
		return nil, e
	}
	defer r.Close()

	l := make([]*tar.Header, 0)
	for {
		hdr, e := r.Next()
		if e == io.EOF {
			return l, nil
		}
		if e != nil {
			return nil, e
		}
		l = append(l, hdr)
	}
	return l, nil
}
