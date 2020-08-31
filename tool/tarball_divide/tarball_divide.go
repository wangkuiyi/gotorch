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
// You can download the MNIST PNG dataset from
// https://github.com/myleott/mnist_png as /tmp/mnist_png.tar.gz and divide it
// using the following commands:
//
// go install ./...
// tarball_divide -out=/tmp /tmp/mnist_png.tar.gz
//
// The above command generates /tmp/[0-9].tar.gz, each of which contains only
// regular image files, no longer the directories.
package main

import (
	"archive/tar"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func main() {
	outDir := flag.String("out", "./", "The output directory")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "Usage: tarball_divide -out=./ input.tar.gz\n")
	}

	if e := divide(flag.Arg(0), *outDir); e != nil {
		log.Fatal(e)
	}
}

func divide(input, output string) error {
	log.Printf("Dividing %s into %s ...\n", input, output)

	oss := make(map[string]*tgz.Writer)
	defer func(oss map[string]*tgz.Writer) {
		for _, w := range oss {
			w.Close()
		}
	}(oss)

	in, e := tgz.OpenFile(input)
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
			return e
		}

		if hdr.Typeflag == tar.TypeReg {
			label := filepath.Base(filepath.Dir(hdr.Name))
			if _, ok := oss[label]; !ok {
				fn := filepath.Join(output, label+".tar.gz")
				log.Println("Creating", fn, label)
				w, e := tgz.CreateFile(fn)
				if e != nil {
					return fmt.Errorf("Cannot create output: %v", e)
				}
				oss[label] = w
			}

			w := oss[label]
			if e := w.WriteHeader(hdr); e != nil {
				return fmt.Errorf("Failed writing header of %s: %v", hdr.Name, e)
			}

			if _, e := io.CopyN(w, in, hdr.Size); e != nil {
				return fmt.Errorf("Failed copy file %s: %v", hdr.Name, e)
			}
		}
	}
	return nil
}
