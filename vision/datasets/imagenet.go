package datasets

import (
	"archive/tar"
	"compress/gzip"
	"image"
	"io"
	"path/filepath"
	"strings"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

type sample struct {
	input torch.Tensor
	label int64
}

// ImageNetLoader provides a convenient interface to reader
// batch from ImageNet dataset.
// Usage:
//
// loader := datasets.ImageNet("/imagenet/train.tar.gz", 32)
// for range loader.Scan() {
//	img, target := imageNet.Minibatch().Data, loader.Minibatch().Target
// }
type ImageNetLoader struct {
	mbSize int
	tr     *tar.Reader
	vocab  map[string]int64 // the vocabulary of labels.
	err    chan error
	ch     chan sample
	trans  *transforms.ComposeTransformer
}

// ImageNet returns ImageNet dataDataLoader
func ImageNet(r io.Reader, vocab map[string]int64, trans *transforms.ComposeTransformer, mbSize int) (*ImageNetLoader, error) {
	tgz, e := newTarGzReader(r)
	if e != nil {
		return nil, e
	}

	l := &ImageNetLoader{
		mbSize: mbSize,
		tr:     tgz,
		err:    make(chan error),
		ch:     make(chan sample, 10), // bufSize=10
		vocab:  vocab,
		trans:  trans,
	}
	go l.read()
	return l, nil
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageNetLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	inputs := make([]torch.Tensor, 0)
	labels := make([]int64, 0)

	for i := 0; i < p.mbSize; i++ {
		s, ok := <-p.ch
		if ok {
			inputs = append(inputs, s.input)
			labels = append(labels, s.label)
		} else {
			break
		}
	}
	return torch.Stack(inputs, 0), torch.NewTensor(labels)
}

func (p *ImageNetLoader) read() {
	defer close(p.ch)
	defer close(p.err)

	for {
		hdr, err := p.tr.Next()
		if err != nil {
			p.err <- err
			break
		}

		if !strings.HasSuffix(strings.ToUpper(hdr.Name), "JPEG") {
			continue
		}

		label := p.vocab[filepath.Base(filepath.Dir(hdr.Name))]

		image, _, err := image.Decode(p.tr)
		if err != nil {
			p.err <- err
			break
		}
		input := p.trans.Run(image)

		p.ch <- sample{input: input.(torch.Tensor), label: label}
	}
}

// Scan return false if no more data
func (p *ImageNetLoader) Scan() bool {
	select {
	case e := <-p.err:
		if e != nil {
			return false
		}
	default:
		return true
	}
	return true
}

// Err returns the error during the scan process, if there is any. io.EOF is not
// considered an error.
func (p *ImageNetLoader) Err() error {
	if e, ok := <-p.err; ok && e != nil && e != io.EOF {
		return e
	}
	return nil
}

// BuildLabelVocabulary returns a vocabulary which mapping from the class name
// to index.  A generica images are arranged in this way:
//
//   blue/xxx.jpeg
//   blue/yyy.jpeg
//   green/zzz.jpeg
//
// this function would scan all sub-directories of root, building an index from
// the class name to it's index.
func BuildLabelVocabulary(reader io.Reader) (map[string]int64, error) {
	tr, e := newTarGzReader(reader)
	if e != nil {
		return nil, e
	}

	classToIdx := make(map[string]int64)
	var idx int64 = 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return classToIdx, nil
		}
		if err != nil {
			return nil, err
		}
		if hdr.FileInfo().Mode().IsRegular() {
			class := filepath.Base(filepath.Dir(hdr.Name))
			if _, ok := classToIdx[class]; !ok {
				classToIdx[class] = idx
				idx++
			}
		}
	}
	return classToIdx, nil
}

func newTarGzReader(r io.Reader) (*tar.Reader, error) {
	// NOTE: gzip.NewReader returns an io.ReadCloser. However, we ignore the
	// chance to call its Close() method, which verifies the checksum, which
	// we don't really care as the data had been consumed by the train loop.
	g, e := gzip.NewReader(r)
	if e != nil {
		return nil, e
	}

	return tar.NewReader(g), nil
}
