package datasets

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"image"
	"image/draw"
	"io"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

// Sample represents a dataset sample which contains
// the image and the class index.
type Sample struct {
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
	batchSize int64
	tr        *tar.Reader
	vocab     map[string]int // the vocabulary of labels.
	isEOF     bool
	samples   []Sample
	trans     *transforms.ComposeTransformer
}

// ImageNet returns ImageNet dataDataLoader
func ImageNet(reader io.Reader, vocab map[string]int, trans *transforms.ComposeTransformer, batchSize int64) (*ImageNetLoader, error) {
	gr, err := gzip.NewReader(reader)
	if err != nil {
		return nil, err
	}
	return &ImageNetLoader{
		batchSize: batchSize,
		tr:        tar.NewReader(gr),
		isEOF:     false,
		vocab:     vocab,
		trans:     trans,
	}, nil
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageNetLoader) Minibatch() Batch {
	// TODO(yancey1989): execute transform function sequentially to transfrom the sample
	// data to Tensors.
	dataArray := []torch.Tensor{}
	labelArray := []torch.Tensor{}
	for _, sample := range p.samples {
		data := p.trans.Run(sample.image)
		label := transforms.ToTensor().Run(sample.target)
		dataArray = append(dataArray, data.(torch.Tensor))
		labelArray = append(labelArray, label)
	}
	return Batch{torch.Stack(dataArray, 0), torch.Stack(labelArray, 0)}
}

func (p *ImageNetLoader) nextSamples() error {
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
		target := p.vocab[filepath.Base(filepath.Dir(hdr.Name))]
		// read image
		data := make([]byte, hdr.Size)
		if _, err := p.tr.Read(data); err != io.EOF {
			return fmt.Errorf("has not read a complete image")
		}
		src, _, err := image.Decode(bytes.NewReader(data))
		m := image.NewRGBA(image.Rect(0, 0, src.Bounds().Dx(), src.Bounds().Dy()))
		draw.Draw(m, m.Bounds(), src, image.ZP, draw.Src)
		p.samples = append(p.samples, Sample{m, target})
	}
	return nil
}

// Scan return false if no more data
func (p *ImageNetLoader) Scan() bool {
	if p.isEOF {
		return false
	}
	must(p.nextSamples())
	if p.isEOF && len(p.samples) == 0 {
		return false
	}
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
