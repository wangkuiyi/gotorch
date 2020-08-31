// This program is supposed to run together with tarball_divide, which divides a
// .tar.gz file into one or more smaller tarball files, where each contains
// files in the same base directory and named by the base directory.  This
// program merge these smaller tarball files back, keeping the property that
// successive files in the merged tarball belong to different base directory.
//
// You can run this program using the following commands:
//
// go install ./...
// tarball_merge -out=/tmp/merged.tar.gz /tmp/[0-9].tar.gz
//
package main

import (
	"flag"
	"fmt"
	"io"
	"log"

	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func main() {
	out := flag.String("out", "./merged.tar.gz", "The output .tar.gz file")
	flag.Parse()

	mergeFiles(flag.Args(), *out)
}

func mergeFiles(fns []string, out string) {
	ins, e := openInputs(fns)
	if e != nil {
		log.Fatal(e)
	}

	w, e := tgz.CreateFile(out)
	if e != nil {
		log.Fatal(e)
	}
	defer w.Close()

	if e := merge(ins, w); e != nil {
		log.Fatal(e)
	}
}

func merge(ins []*tgz.Reader, w *tgz.Writer) error {
	closed := 0
	for closed < len(ins) {
		for i, r := range ins {
			if r != nil {
				hdr, e := r.Next()
				if e == io.EOF {
					r.Close()
					closed++
					ins[i] = nil // Mark invalid
				} else if e != nil {
					return fmt.Errorf("Reading tarball: %v", e)
				}

				if hdr == nil {
					continue // Undocumented must-handle case.
				}

				if e := w.WriteHeader(hdr); e != nil {
					return fmt.Errorf("Failed writing header of %s: %v", hdr.Name, e)
				}

				if _, e := io.CopyN(w, r, hdr.Size); e != nil {
					return fmt.Errorf("Failed copy file %s: %v", hdr.Name, e)
				}
			}
		}
	}
	return nil
}

func openInputs(fns []string) ([]*tgz.Reader, error) {
	in := make([]*tgz.Reader, 0)
	for _, fn := range fns {
		r, e := tgz.OpenFile(fn)
		if e != nil {
			return nil, fmt.Errorf("Cannot open %s: %v", fn, e)
		}
		in = append(in, r)
	}
	if len(in) <= 0 {
		return nil, fmt.Errorf("No input file specified")
	}
	return in, nil
}
