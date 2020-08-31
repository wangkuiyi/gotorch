package tgz

import (
	"archive/tar"
	"io"
	"io/ioutil"
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
		switch {
		case e == io.EOF:
			return l, nil
		case e != nil:
			return nil, e
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
