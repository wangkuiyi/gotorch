package tgz

import (
	"archive/tar"
	"io"
	"io/ioutil"
)

// ListFile list regular files in a .tar.gz file.
func ListFile(fn string) ([]*tar.Header, error) {
	r, e := OpenFile(fn)
	if e != nil {
		return nil, e
	}
	defer r.Close()

	return List(r)
}

// List reads and lists entries read from a reader.
func List(r *Reader) ([]*tar.Header, error) {
	l := make([]*tar.Header, 0)
	for {
		hdr, e := r.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, e
		}

		if hdr == nil {
			continue
		}

		switch hdr.Typeflag {
		case tar.TypeDir:
		case tar.TypeReg:
			l = append(l, hdr)
			io.Copy(ioutil.Discard, r)
		}
	}
	return l, nil
}
