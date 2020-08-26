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

// sample represents a dataset sample which contains
// the image and the class index.
type sample struct {
	image  image.Image
	target int
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
	err    error
	inputs []torch.Tensor // inputs and labels form a minibatch.
	labels []int64
	trans  *transforms.ComposeTransformer
}

// ImageNet returns ImageNet dataDataLoader
func ImageNet(r io.Reader, vocab map[string]int64, trans *transforms.ComposeTransformer, mbSize int) (*ImageNetLoader, error) {
	tgz, e := newTarGzReader(r)
	if e != nil {
		return nil, e
	}
	return &ImageNetLoader{
		mbSize: mbSize,
		tr:     tgz,
		err:    nil,
		vocab:  vocab,
		trans:  trans,
	}, nil
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageNetLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	return torch.Stack(p.inputs, 0), torch.NewTensor(p.labels)
}

func (p *ImageNetLoader) retreiveMinibatch() {
	p.inputs = []torch.Tensor{}
	p.labels = []int64{}
	for {
		hdr, err := p.tr.Next()
		if err != nil {
			p.err = err
			break
		}
		if !strings.HasSuffix(strings.ToUpper(hdr.Name), "JPEG") {
			continue
		}

		label := p.vocab[filepath.Base(filepath.Dir(hdr.Name))]
		p.labels = append(p.labels, label)

		image, _, err := image.Decode(p.tr)
		if err != nil {
			p.err = err
			break
		}
		input := p.trans.Run(image)
		p.inputs = append(p.inputs, input.(torch.Tensor))

		if len(p.inputs) == p.mbSize {
			break
		}
	}
}

// Scan return false if no more data
func (p *ImageNetLoader) Scan() bool {
	if p.err == io.EOF {
		return false
	}
	p.retreiveMinibatch()
	if p.err != nil && len(p.inputs) == 0 {
		return false
	}
	return true
}

// Err returns the error during the scan process, if there is any. io.EOF is not
// considered an error.
func (p *ImageNetLoader) Err() error {
	if p.err == io.EOF {
		return nil
	}
	return p.err
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
