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

	ins, e := openInputs()
	if e != nil {
		log.Fatal(e)
	}

	w, e := tgz.CreateFile(*out)
	if e != nil {
		log.Fatal(e)
	}

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

func openInputs() ([]*tgz.Reader, error) {
	in := make([]*tgz.Reader, 0)
	for _, fn := range flag.Args() {
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
