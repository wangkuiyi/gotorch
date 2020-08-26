package datasets

import (
	"archive/tar"
	"compress/gzip"
	"image"
	"image/draw"
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
	mbSize  int
	tr      *tar.Reader
	vocab   map[string]int // the vocabulary of labels.
	eof     bool
	samples []sample
	trans   *transforms.ComposeTransformer
}

// ImageNet returns ImageNet dataDataLoader
func ImageNet(r io.Reader, vocab map[string]int, trans *transforms.ComposeTransformer, mbSize int) (*ImageNetLoader, error) {
	tgz, e := newTarGzReader(r)
	if e != nil {
		return nil, e
	}
	return &ImageNetLoader{
		mbSize: mbSize,
		tr:     tgz,
		eof:    false,
		vocab:  vocab,
		trans:  trans,
	}, nil
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageNetLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	// TODO(yancey1989): execute transform function sequentially to transfrom the sample
	// data to Tensors.
	dataArray := []torch.Tensor{}
	var labelArray []int64
	for _, sample := range p.samples {
		data := p.trans.Run(sample.image)
		dataArray = append(dataArray, data.(torch.Tensor))
		labelArray = append(labelArray, int64(sample.target))
	}
	return torch.Stack(dataArray, 0), torch.NewTensor(labelArray)
}

func (p *ImageNetLoader) nextSamples() error {
	p.samples = []sample{}
	for {
		hdr, err := p.tr.Next()
		if err == io.EOF {
			p.eof = true
			break
		}
		if err != nil {
			return err
		}
		if !strings.HasSuffix(strings.ToUpper(hdr.Name), "JPEG") {
			continue
		}
		// read target
		target := p.vocab[filepath.Base(filepath.Dir(hdr.Name))]
		src, _, err := image.Decode(p.tr)
		if err != nil {
			return err
		}
		m := image.NewRGBA(image.Rect(0, 0, src.Bounds().Dx(), src.Bounds().Dy()))
		draw.Draw(m, m.Bounds(), src, image.ZP, draw.Src)
		p.samples = append(p.samples, sample{m, target})
		if len(p.samples) == p.mbSize {
			break
		}
	}
	return nil
}

// Scan return false if no more data
func (p *ImageNetLoader) Scan() bool {
	if p.eof {
		return false
	}
	must(p.nextSamples())
	if p.eof && len(p.samples) == 0 {
		return false
	}
	return true
}

func must(e error) {
	if e != nil {
		panic(e)
	}
}

// BuildLabelVocabulary returns a vocabulary which mapping from the class name to index.
// A generica images are arranged in this way:
//
//   blue/xxx.jpeg
//   blue/yyy.jpeg
//   green/zzz.jpeg
//
// this function would scan all sub-directories of root, building an index from the class name
// to it's index.
func BuildLabelVocabulary(reader io.Reader) (map[string]int, error) {
	tr, e := newTarGzReader(reader)
	if e != nil {
		return nil, e
	}

	classToIdx := make(map[string]int)
	idx := 0
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
