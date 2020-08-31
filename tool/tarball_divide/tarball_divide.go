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
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/wangkuiyi/gotorch/tool/tgz"
)

func main() {
	outDir := flag.String("out", "./", "The output directory")
	flag.Parse()

	log.Println(flag.NArg(), flag.Arg(0))

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
		// if e == io.EOF {
		// 	log.Println("EOF")
		// 	break
		// }
		// if e != nil {
		// 	return fmt.Errorf("Failed reading input: %v", e)
		// }

		// log.Printf("%+v\n", hdr)

		// if hdr.Typeflag == tar.TypeReg {
		// 	log.Println("Is regular file")

		// 	label := filepath.Base(filepath.Dir(hdr.Name))
		// 	log.Println(hdr.Name, label)
		// 	if _, ok := oss[label]; !ok {
		// 		fn := filepath.Join(output, label+".tar.gz")
		// 		log.Println("Creating", fn, label)
		// 		w, e := tgz.CreateFile(fn)
		// 		if e != nil {
		// 			return fmt.Errorf("Cannot create output: %v", e)
		// 		}
		// 		oss[label] = w
		// 	}

		// 	w := oss[label]
		// 	if e := w.WriteHeader(hdr); e != nil {
		// 		return fmt.Errorf("Failed writing header of %s: %v", hdr.Name, e)
		// 	}

		// 	if _, e := io.CopyN(w, in, hdr.Size); e != nil {
		// 		return fmt.Errorf("Failed copy file %s: %v", hdr.Name, e)
		// 	}
		// } else {
		// 	log.Println("Not regular file")
		// 	_, e := io.Copy(ioutil.Discard, in) // Discard or r.Next() returns EOF.
		// 	log.Println(e)
		// }
		log.Println(e, hdr)
		if e == io.EOF {
			break
		}
	}
	return nil
}
