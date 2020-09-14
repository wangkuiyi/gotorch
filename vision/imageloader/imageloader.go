package imageloader

import (
	"image"
	"io"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

type sample struct {
	data  image.Image
	label int
}

// ImageLoader struct
type ImageLoader struct {
	r         *tgz.Reader
	vocab     map[string]int
	samples   chan sample
	errChan   chan error
	err       error
	trans1    *transforms.ComposeTransformer // transforms before `ToTensor`
	trans2    *transforms.ComposeTransformer // transforms after and include `ToTensor`
	mbSize    int
	inputs    []torch.Tensor
	labels    []int64
	pinMemory bool
}

// New returns an ImageLoader
func New(fn string, vocab map[string]int, trans *transforms.ComposeTransformer,
	mbSize int, pinMemory bool) (*ImageLoader, error) {
	r, e := tgz.OpenFile(fn)
	if e != nil {
		return nil, e
	}
	trans1, trans2 := splitComposeByToTensor(trans)
	m := &ImageLoader{
		r:         r,
		vocab:     vocab,
		samples:   make(chan sample, mbSize*4),
		errChan:   make(chan error, 1),
		trans1:    trans1,
		trans2:    trans2,
		mbSize:    mbSize,
		pinMemory: pinMemory,
	}
	go m.read()
	return m, nil
}
func (p *ImageLoader) tensorGC() {
	p.inputs = []torch.Tensor{}
	p.labels = []int64{}
	torch.GC()
}

// Scan return false if no more data
func (p *ImageLoader) Scan() bool {
	p.tensorGC()
	select {
	case e := <-p.errChan:
		if e != nil && e != io.EOF {
			p.err = e
			return false
		}
	default:
		// no error received
	}
	return p.retreiveMinibatch()
}

func (p *ImageLoader) read() {
	defer close(p.samples)
	defer close(p.errChan)
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
		p.samples <- sample{p.trans1.Run(m).(image.Image), label}
	}
}

func (p *ImageLoader) retreiveMinibatch() bool {
	for i := 0; i < p.mbSize; i++ {
		sample, ok := <-p.samples
		if ok {
			p.inputs = append(p.inputs, p.trans2.Run(sample.data).(torch.Tensor))
			p.labels = append(p.labels, int64(sample.label))
		} else {
			if i == 0 {
				return false
			}
			return true
		}
	}
	return true
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	i := torch.Stack(p.inputs, 0)
	l := torch.NewTensor(p.labels)
	if p.pinMemory {
		return i.PinMemory(), l.PinMemory()
	}
	return i, l
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
