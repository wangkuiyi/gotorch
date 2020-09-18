package imageloader

import (
	"image"
	"io"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

type miniBatch struct {
	data  torch.Tensor
	label torch.Tensor
}

// ImageLoader struct
type ImageLoader struct {
	r         *tgz.Reader
	vocab     map[string]int
	mbChan    chan miniBatch
	errChan   chan error
	err       error
	trans     *transforms.ComposeTransformer // transforms before `ToTensor`
	mbSize    int
	input     torch.Tensor
	label     torch.Tensor
	pinMemory bool
}

// New returns an ImageLoader
func New(fn string, vocab map[string]int, trans *transforms.ComposeTransformer,
	mbSize int, pinMemory bool) (*ImageLoader, error) {
	r, e := tgz.OpenFile(fn)
	if e != nil {
		return nil, e
	}
	m := &ImageLoader{
		r:         r,
		vocab:     vocab,
		mbChan:    make(chan miniBatch, 4),
		errChan:   make(chan error, 1),
		trans:     trans,
		mbSize:    mbSize,
		pinMemory: pinMemory,
	}
	go m.read()
	return m, nil
}

// Scan return false if no more data
func (p *ImageLoader) Scan() bool {
	torch.GC()
	select {
	case e := <-p.errChan:
		if e != nil && e != io.EOF {
			p.err = e
			return false
		}
	default:
		// no error received
	}
	if miniBatch, ok := <-p.mbChan; ok {
		p.input = miniBatch.data
		p.label = miniBatch.label
		return true
	}
	torch.FinishGC()
	return false
}

func (p *ImageLoader) read() {
	inputs := []torch.Tensor{}
	labels := []int64{}

	defer func() {
		close(p.mbChan)
		close(p.errChan)
	}()
	for {
		hdr, err := p.r.Next()
		if err != nil {
			p.errChan <- err
			break
		}
		if !hdr.FileInfo().Mode().IsRegular() {
			continue
		}
		classStr := filepath.Base(filepath.Dir(hdr.Name))
		label := p.vocab[classStr]
		m, _, err := image.Decode(p.r)
		if err != nil {
			p.errChan <- err
			break
		}
		inputs = append(inputs, p.trans.Run(m).(torch.Tensor))
		labels = append(labels, int64(label))
		if len(inputs) == p.mbSize {
			d := torch.Stack(inputs, 0)
			l := torch.NewTensor(labels)
			p.mbChan <- miniBatch{d, l}
			inputs = []torch.Tensor{}
			labels = []int64{}
		}
	}
	if len(inputs) != 0 {
		p.mbChan <- miniBatch{torch.Stack(inputs, 0), torch.NewTensor(labels)}
	}
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	return p.input, p.label
}

// Err returns the error during the scan process, if there is any. io.EOF is not
// considered an error.
func (p *ImageLoader) Err() error {
	return p.err
}

// BuildLabelVocabularyFromTgz build a label vocabulary from the image tgz file
func BuildLabelVocabularyFromTgz(fn string) (map[string]int, error) {
	vocab := make(map[string]int)
	l, e := tgz.ListFile(fn)
	if e != nil {
		return nil, e
	}
	idx := 0
	for _, hdr := range l {
		class := filepath.Base(filepath.Dir(hdr.Name))
		if _, ok := vocab[class]; !ok {
			vocab[class] = idx
			idx++
		}
	}
	return vocab, nil
}

func splitComposeByToTensor(compose *transforms.ComposeTransformer) (*transforms.ComposeTransformer, *transforms.ComposeTransformer) {
	idx := len(compose.Transforms)
	for i, trans := range compose.Transforms {
		if _, ok := trans.(*transforms.ToTensorTransformer); ok {
			idx = i
			break
		}
	}
	return transforms.Compose(compose.Transforms[:idx]...), transforms.Compose(compose.Transforms[idx:]...)
}
