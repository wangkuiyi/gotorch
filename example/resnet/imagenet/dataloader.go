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
	batchSize  int64
	tr         *tar.Reader
	classToIdx map[string]int
	isEOF      bool
	samples    []Sample
}

// NewDataLoader returns ImageNet dataDataLoader
func NewDataLoader(reader io.Reader, batchSize int64) (*DataLoader, error) {
	// using io.TeeReader to read the input io.Reader twice, the first reading
	// make a mapping from class name to target index, the second read all images.
	var buff bytes.Buffer
	r := io.TeeReader(reader, &buff)
	classToIdx, err := makeClassToIdx(r)
	if err != nil {
		return nil, err
	}

	gr, err := gzip.NewReader(&buff)
	if err != nil {
		return nil, err
	}
	return &DataLoader{
		batchSize:  batchSize,
		tr:         tar.NewReader(gr),
		isEOF:      false,
		classToIdx: classToIdx,
	}, nil
}

// Batch returns data and target Tensor
func (p *DataLoader) Batch() (torch.Tensor, torch.Tensor) {
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
		target := p.classToIdx[filepath.Base(filepath.Dir(hdr.Name))]
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

func makeClassToIdx(reader io.Reader) (map[string]int, error) {
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
