package imageloader

import (
	"fmt"
	"image"
	"io"
	"path/filepath"
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
	"gocv.io/x/gocv"
)

type miniBatch struct {
	data  torch.Tensor
	label torch.Tensor
}

// RGB color
const RGB string = "rgb"

// GRAY color
const GRAY string = "gray"

// ImageLoader struct
type ImageLoader struct {
	r          *tgz.Reader
	vocab      map[string]int
	mbChan     chan miniBatch
	errChan    chan error
	err        error
	trans1     *transforms.ComposeTransformer // transforms before `ToTensor`
	trans2     *transforms.ComposeTransformer // transforms after and include `ToTensor`
	mbSize     int
	input      torch.Tensor
	label      torch.Tensor
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
		mbChan:     make(chan miniBatch, 4),
		errChan:    make(chan error, 1),
		trans1:     trans1,
		trans2:     trans2,
		mbSize:     mbSize,
		pinMemory:  pinMemory,
		colorSpace: colorSpace,
	}
	runtime.LockOSThread()
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
	inputs := []gocv.Mat{}
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
		buffer := make([]byte, hdr.Size)
		io.ReadFull(p.r, buffer)
		m, err := readImage(buffer, p.colorSpace)
		if err != nil {
			p.errChan <- err
			break
		}
		if m.Empty() {
			panic("read invalid image content!")
		}
		inputs = append(inputs, p.trans1.Run(m).(gocv.Mat))
		labels = append(labels, int64(label))
		if len(inputs) == p.mbSize {
			p.mbChan <- p.collateMiniBatch(inputs, labels)
			inputs = []gocv.Mat{}
			labels = []int64{}
		}
	}
	if len(inputs) > 0 {
		p.mbChan <- p.collateMiniBatch(inputs, labels)
	}
}

func (p *ImageLoader) collateMiniBatch(inputs []gocv.Mat, labels []int64) miniBatch {
	w := inputs[0].Cols()
	h := inputs[0].Rows()
	blob := gocv.NewMat()
	defer blob.Close()
	gocv.BlobFromImages(inputs, &blob, 1.0/255.0, image.Pt(w, h), gocv.NewScalar(0, 0, 0, 0), false, false, gocv.MatTypeCV32F)
	i := p.trans2.Run(blob).(torch.Tensor)
	l := torch.NewTensor(labels)
	if p.pinMemory {
		return miniBatch{i.PinMemory(), l.PinMemory()}
	}
	return miniBatch{i, l}
}

// Minibatch returns a minibatch with data and label Tensor
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

func readImage(buffer []byte, colorSpace string) (gocv.Mat, error) {
	var m gocv.Mat
	var e error
	if colorSpace == RGB {
		m, e = gocv.IMDecode(buffer, gocv.IMReadColor)
	} else if colorSpace == GRAY {
		m, e = gocv.IMDecode(buffer, gocv.IMReadGrayScale)
	} else {
		return m, fmt.Errorf("Cannot read image with color space %v", colorSpace)
	}
	if !m.Empty() && colorSpace == RGB {
		gocv.CvtColor(m, &m, gocv.ColorBGRToRGB)
	}
	return m, e
}
