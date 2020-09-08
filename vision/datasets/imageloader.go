package datasets

import (
	"image"
	"io"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

// ImageLoader struct
type ImageLoader struct {
	r      *tgz.Reader
	vocab  map[string]int64
	err    error
	inputs []torch.Tensor
	labels []int64
	trans  *transforms.ComposeTransformer
	mbSize int
}

func (p *ImageLoader) tensorGC() {
	p.inputs = []torch.Tensor{}
	p.labels = []int64{}
	torch.GC()
}

// NewImageLoader returns an ImageLoader
func NewImageLoader(fn string, vocab map[string]int64, trans *transforms.ComposeTransformer, mbSize int) (*ImageLoader, error) {
	r, e := tgz.OpenFile(fn)
	if e != nil {
		return nil, e
	}
	return &ImageLoader{
		r:      r,
		vocab:  vocab,
		err:    nil,
		trans:  trans,
		mbSize: mbSize,
	}, nil
}

// Scan return false if no more dat
func (p *ImageLoader) Scan() bool {
	p.tensorGC()
	if p.err != nil {
		p.r.Close()
		return false
	}
	p.retreiveMinibatch()
	return p.err == nil || p.err == io.EOF // the next call will return false
}

func (p *ImageLoader) retreiveMinibatch() {
	for {
		hdr, err := p.r.Next()
		if err != nil {
			p.err = err
			break
		}
		if !hdr.FileInfo().Mode().IsRegular() {
			continue
		}
		classStr := filepath.Base(filepath.Dir(hdr.Name))
		label := p.vocab[classStr]
		p.labels = append(p.labels, label)

		m, _, err := image.Decode(p.r)
		if err != nil {
			p.err = err
			break
		}
		input := p.trans.Run(m)
		p.inputs = append(p.inputs, input.(torch.Tensor))

		if len(p.inputs) == p.mbSize {
			break
		}
	}
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	return torch.Stack(p.inputs, 0), torch.NewTensor(p.labels)
}

// Err returns the error during the scan process, if there is any. io.EOF is not
// considered an error.
func (p *ImageLoader) Err() error {
	if p.err == io.EOF {
		return nil
	}
	return p.err
}

// BuildLabelVocabularyFromTgz build a label vocabulary from the image tgz file
func BuildLabelVocabularyFromTgz(fn string) (map[string]int64, error) {
	vocab := make(map[string]int64)
	l, e := tgz.ListFile(fn)
	if e != nil {
		return nil, e
	}
	idx := 0
	for _, hdr := range l {
		class := filepath.Base(filepath.Dir(hdr.Name))
		if _, ok := vocab[class]; !ok {
			vocab[class] = int64(idx)
			idx++
		}
	}
	return vocab, nil
}
