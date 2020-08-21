package imagenet

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
)

// Sample represents a dataset sample which contains
// the image and the class index.
type Sample struct {
	image  image.Image
	target int
}

// Loader provides a convenient interface to reader
// batch from ImageNet dataset.
// Usage:
//
// transforms:= []Transform{ToTensor{}, RandomResizedCrop{}})
// loader := imagenet.Loader("/imagenet/train.tar.gz",  32, transforms)
// for range imageNet.Scan() {
//	img, target := imageNet.Batch()
// }
type Loader struct {
	batchSize  int64
	transforms []Transform
	f          *os.File
	tr         *tar.Reader
	classToIdx map[string]int
	isEOF      bool
	samples    []Sample
}

// Transform interface
type Transform interface {
	Do(interface{}) torch.Tensor
}

// NewLoader returns ImageNet dataloader
func NewLoader(tarFile string, batchSize int64, transforms []Transform) (*Loader, error) {
	classToIdx, err := makeClassToIdx(tarFile)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(tarFile)
	if err != nil {
		return nil, err
	}
	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	return &Loader{
		batchSize:  batchSize,
		transforms: transforms,
		f:          f,
		tr:         tar.NewReader(gr),
		isEOF:      false,
		classToIdx: classToIdx,
	}, nil
}

// Close this loader
func (p *Loader) Close() error {
	return p.f.Close()
}

// Batch returns data and target Tensor
func (p *Loader) Batch() (torch.Tensor, torch.Tensor) {
	return torch.RandN([]int64{p.batchSize, 3, 2}, false), torch.RandN([]int64{p.batchSize, 1}, false)
}

func (p *Loader) updateSamples() error {
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
		target := p.classToIdx[filepath.Base(filepath.Dir(hdr.Name))]
		// read image
		data := make([]byte, hdr.Size)
		_, err = p.tr.Read(data)
		// TODO(yancey1989): supports reading very large image as streaming
		if err != io.EOF {
			return fmt.Errorf("has not read the complete image")
		}
		image, _, err := image.Decode(bytes.NewReader(data))
		p.samples = append(p.samples, Sample{image, target})
	}
	return nil
}

// Scan return false if no more data
func (p *Loader) Scan() bool {
	if p.isEOF {
		return false
	}
	must(p.updateSamples())
	return true
}

func must(e error) {
	if e != nil {
		panic(e)
	}
}

func makeClassToIdx(tarFile string) (map[string]int, error) {
	f, err := os.Open(tarFile)
	if err != nil {
		return nil, err
	}
	gr, err := gzip.NewReader(f)
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
