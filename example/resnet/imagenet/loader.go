package imagenet

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"image"
	"image/draw"
	"io"
	"path/filepath"
	"unsafe"

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
	batchSize int64
	tr        *tar.Reader
	vocab     map[string]int // the vocabulary of labels.
	isEOF     bool
	samples   []Sample
}

// NewDataLoader returns ImageNet dataDataLoader
func NewDataLoader(reader io.Reader, vocab map[string]int, batchSize int64) (*DataLoader, error) {
	gr, err := gzip.NewReader(reader)
	if err != nil {
		return nil, err
	}
	return &DataLoader{
		batchSize: batchSize,
		tr:        tar.NewReader(gr),
		isEOF:     false,
		vocab:     vocab,
	}, nil
}

// Minibatch returns a minibash with data and label Tensor
func (p *DataLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	// TODO(yancey1989): execute transform function sequentially to transfrom the sample
	// data to Tensors.
	dataArray := []torch.Tensor{}
	labelArray := []torch.Tensor{}
	for _, sample := range p.samples {
		data, err := ToTensor(sample.image)
		must(err)
		label, err := ToTensor(sample.target)
		must(err)
		dataArray = append(dataArray, data)
		labelArray = append(labelArray, label)
	}
	return torch.Stack(dataArray, 0), torch.Stack(labelArray, 0)
	//return torch.RandN([]int64{p.batchSize, 3, 2}, false), torch.RandN([]int64{p.batchSize, 1}, false)
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
func (p *DataLoader) Scan() bool {
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

// ToTensor returns a torch.Tensor from the given object
func ToTensor(obj interface{}) (torch.Tensor, error) {
	switch v := obj.(type) {
	case image.Image:
		return imageToTensor(obj.(image.Image)), nil
	case int:
		return intToTensor(obj.(int)), nil
	default:
		return torch.Tensor{}, fmt.Errorf("ToTensor transform does not support type: %T", v)
	}
}

func imageToTensor(img image.Image) torch.Tensor {
	width, height := img.Bounds().Max.X, img.Bounds().Max.Y
	array := make([][][3]float32, height)

	for x := 0; x < height; x++ {
		row := make([][3]float32, width)
		for y := 0; y < width; y++ {
			c := img.(*image.RGBA).RGBAAt(x, y)
			row[y] = [3]float32{float32(c.R / 255.0), float32(c.G / 255.0), float32(c.B / 255.0)}
		}
		array[x] = row
	}
	return torch.FromBlob(unsafe.Pointer(&array[0][0][0]), torch.Float, []int64{int64(width), int64(height), 3})
}

func intToTensor(x int) torch.Tensor {
	array := make([]int, 1)
	array[0] = x
	return torch.FromBlob(unsafe.Pointer(&array[0]), torch.Int, []int64{1})
}
