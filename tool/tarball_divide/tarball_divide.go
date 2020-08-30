// tarball_divide takes a .tar.gz file of images and outputs one or more .tar.gz
// of images, where each contains images in a certain base directory.  For
// example, the input tarball might contain three image files:
//
//  another_dir/dog/b.png
//  Yet_another_dir/dog/c.png
//  some_dir/0/a.png
//
// The output includes two tarballs
//
// - 0.tar.gz includes one file
//   - some_dir/0/a.png
//
// - dog.tar.gz inlcudes
//   - another_dir/dog/b.png
//   - Yet_another_dir/dog/c.png
//
// tarball_merge can then merge the two tarballs into one, with an interleaved
// order of images:
//
//  another_dir/dog/b.png
//  some_dir/0/a.png
//  Yet_another_dir/dog/c.png
//
// We need this pair of tools because the training of image classification
// models often needs to read images from a container file, the tarball, and we
// want each minibatch of successive images belong to different lables.  In the
// convention, the base directory name is the label.
//
package main

import (
	"archive/tar"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func main() {
	outDir := flag.String("out", "./", "The output directory")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "Usage: tarball_divide input.tar.gz -out=./")
	}

	divide(flag.Arg(0), *outDir)
}

func divide(input, output string) error {
	oss := make(map[string]*TarGzWriter)
	defer func(oss map[string]*TarGzWriter) {
		for _, w := range oss {
			w.Close()
		}
	}(oss)

	in, e := OpenTarGzFile(input)
	if e != nil {
		return fmt.Errorf("Cannot create reader: %v", e)
	}
	defer in.Close()

	for {
		hdr, e := in.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			return fmt.Errorf("Failed reading input: %v", e)
		}

		label := filepath.Base(filepath.Dir(hdr.Name))
		if _, ok := oss[label]; !ok {
			w, e := CreatTarGzFile(filepath.Join(output, label+".tar.gz"))
			if e != nil {
				return fmt.Errorf("Cannot create output: %v", e)
			}
			oss[label] = w
		}

		w := oss[label]
		if e := w.WriteHeader(hdr); e != nil {
			return fmt.Errorf("Failed writing header: %v", e)
		}

		if _, e := io.CopyN(w, in, hdr.Size); e != nil {
			return fmt.Errorf("Failed copy file content: %v", e)
		}
	}
	return nil
}

// TarGzReader includes a tar reader and its underlying readers.
type TarGzReader struct {
	*tar.Reader
	g *gzip.Reader
	f *os.File
}

// OpenTarGzFile returns a TarGzReader
func OpenTarGzFile(fn string) (*TarGzReader, error) {
	f, e := os.Open(fn)
	if e != nil {
		return nil, e
	}

	g, e := gzip.NewReader(f)
	if e != nil {
		return nil, e
	}

	return &TarGzReader{
		Reader: tar.NewReader(g),
		g:      g,
		f:      f,
	}, nil
}

// Close calls gzip.Reader.Close() and os.File.Close if the underlying storage
// is a file.
func (r *TarGzReader) Close() error {
	if e := r.g.Close(); e != nil {
		return e
	}
	if r.f != nil {
		return r.f.Close()
	}
	return nil
}

// TarGzWriter defines a writer to a .tar.gz file.
type TarGzWriter struct {
	*tar.Writer
	g *gzip.Writer
	f *os.File
}

// CreatTarGzFile creates a writer.
func CreatTarGzFile(fn string) (*TarGzWriter, error) {
	f, e := os.Create(fn)
	if e != nil {
		return nil, e
	}

	g := gzip.NewWriter(f)
	return &TarGzWriter{
		Writer: tar.NewWriter(g),
		g:      g,
		f:      f}, nil
}

// Close calls the tar.Writer.Close(), gzip.Writer.Close(), and if the
// underlying storage is a file, os.File.Close().
func (w *TarGzWriter) Close() error {
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

// ListTarGzFile list contents in a .tar.gz file.
func ListTarGzFile(fn string) ([]*tar.Header, error) {
	r, e := OpenTarGzFile(fn)
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
