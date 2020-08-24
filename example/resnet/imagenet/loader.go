package imagenet

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"image"
	"io"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
)

// Sample represents a dataset sample which contains
// the image and the class index.
type Sample struct {
	image  image.Image
	target int
}

// DataLoader provides a convenient interface to reader
// batch from ImageNet dataset.
// Usage:
//
// DataLoader := imagenet.DataLoader("/imagenet/train.tar.gz", 32)
// for range imageNet.Scan() {
//	img, target := imageNet.Batch()
// }
type DataLoader struct {
	batchSize       int64
	tr              *tar.Reader
	labelVocabulary map[string]int
	isEOF           bool
	samples         []Sample
}

// NewDataLoader returns ImageNet dataDataLoader
func NewDataLoader(reader io.Reader, labelVob map[string]int, batchSize int64) (*DataLoader, error) {
	gr, err := gzip.NewReader(reader)
	if err != nil {
		return nil, err
	}
	return &DataLoader{
		batchSize:       batchSize,
		tr:              tar.NewReader(gr),
		isEOF:           false,
		labelVocabulary: labelVob,
	}, nil
}

// Minibatch returns a minibash with data and label Tensor
func (p *DataLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	// TODO(yancey1989): execute transform function sequentially to transfrom the sample
	// data to Tensors.
	return torch.RandN([]int64{p.batchSize, 3, 2}, false), torch.RandN([]int64{p.batchSize, 1}, false)
}

func (p *DataLoader) nextSamples() error {
	p.samples = []Sample{}
	for i := int64(0); i < p.batchSize; i++ {
		hdr, err := p.tr.Next()
		if err == io.EOF {
			p.isEOF = true
			break
		}
		if err != nil {
			return err
		}
		// read target
		target := p.labelVocabulary[filepath.Base(filepath.Dir(hdr.Name))]
		// read image
		data := make([]byte, hdr.Size)
		if _, err := p.tr.Read(data); err != io.EOF {
			return fmt.Errorf("has not read a complete image")
		}
		image, _, err := image.Decode(bytes.NewReader(data))
		p.samples = append(p.samples, Sample{image, target})
	}
	return nil
}

// Scan return false if no more data
func (p *DataLoader) Scan() bool {
	if p.isEOF {
		return false
	}
	must(p.nextSamples())
	return true
}

func must(e error) {
	if e != nil {
		panic(e)
	}
}

// BuildLabelVocabulary returns a vocabulary which mapping from the class name to index
func BuildLabelVocabulary(reader io.Reader) (map[string]int, error) {
	gr, err := gzip.NewReader(reader)
	if err != nil {
		return nil, err
	}
	tr := tar.NewReader(gr)
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
		class := filepath.Base(filepath.Dir(hdr.Name))
		if _, ok := classToIdx[class]; !ok {
			classToIdx[class] = idx
			idx++
		}
	}
	return classToIdx, nil
}
