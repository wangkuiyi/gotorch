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
	return torch.Stack(dataArray, 0), torch.NewTensor([]int64{1, 2, 3, 4})
}

func (p *ImageNetLoader) nextSamples() error {
	p.samples = []Sample{}
	for {
		hdr, err := p.tr.Next()
		if err == io.EOF {
			p.isEOF = true
			break
		}
		if err != nil {
			return err
		}
		if !strings.HasSuffix(strings.ToUpper(hdr.Name), "JPEG") || strings.HasPrefix(filepath.Base(hdr.Name), "._") {
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
		p.samples = append(p.samples, Sample{m, target})
		if int64(len(p.samples)) == p.batchSize {
			break
		}
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
		if hdr.FileInfo().IsDir() {
			class := filepath.Base(filepath.Dir(hdr.Name))
			if _, ok := classToIdx[class]; !ok {
				classToIdx[class] = idx
				idx++
			}
		}
	}
	return classToIdx, nil
}
