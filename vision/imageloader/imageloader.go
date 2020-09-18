package imageloader

import (
	"image"
	"io"
	"log"
	"path/filepath"

	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
	"gocv.io/x/gocv"
)

// RGB color
const RGB string = "rgb"

// GRAY color
const GRAY string = "gray"

type sample struct {
	data  gocv.Mat
	label int
}

// ImageLoader struct
type ImageLoader struct {
	r          *tgz.Reader
	vocab      map[string]int
	samples    chan sample
	errChan    chan error
	err        error
	trans1     *transforms.ComposeTransformer // transforms before `ToTensor`
	trans2     *transforms.ComposeTransformer // transforms after and include `ToTensor`
	mbSize     int
	inputs     []gocv.Mat
	labels     []int64
	pinMemory  bool
	colorSpace string
}

// New returns an ImageLoader
func New(fn string, vocab map[string]int, trans *transforms.ComposeTransformer,
	mbSize int, pinMemory bool, colorSpace string) (*ImageLoader, error) {
	r, e := tgz.OpenFile(fn)
	if e != nil {
		return nil, e
	}
	trans1, trans2 := splitComposeByToTensor(trans)
	m := &ImageLoader{
		r:          r,
		vocab:      vocab,
		samples:    make(chan sample, mbSize*4),
		errChan:    make(chan error, 1),
		trans1:     trans1,
		trans2:     trans2,
		mbSize:     mbSize,
		pinMemory:  pinMemory,
		colorSpace: colorSpace,
	}
	go m.read()
	return m, nil
}
func (p *ImageLoader) tensorGC() {
	for _, input := range p.inputs {
		input.Close()
	}
	p.inputs = []gocv.Mat{}
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
		buffer := make([]byte, hdr.Size)
		p.r.Read(buffer)
		var m gocv.Mat
		if p.colorSpace == RGB {
			m, err = gocv.IMDecode(buffer, gocv.IMReadColor)
		} else if p.colorSpace == GRAY {
			m, err = gocv.IMDecode(buffer, gocv.IMReadGrayScale)
		} else {
			log.Fatalf("Cannot read image with color space %v", p.colorSpace)
		}
		if err != nil {
			p.errChan <- err
			break
		}
		if m.Empty() {
			continue
		}
		if p.colorSpace == RGB {
			gocv.CvtColor(m, &m, gocv.ColorBGRToRGB)
		}
		p.samples <- sample{p.trans1.Run(m).(gocv.Mat), label}
	}
}

func (p *ImageLoader) retreiveMinibatch() bool {
	for i := 0; i < p.mbSize; i++ {
		sample, ok := <-p.samples
		if ok {
			p.inputs = append(p.inputs, sample.data)
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
	w := p.inputs[0].Cols()
	h := p.inputs[0].Rows()
	blob := gocv.NewMat()
	defer blob.Close()
	gocv.BlobFromImages(p.inputs, &blob, 1.0/255.0, image.Pt(w, h), gocv.NewScalar(0, 0, 0, 0), false, false, gocv.MatTypeCV32F)
	i := p.trans2.Run(blob).(torch.Tensor)
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
