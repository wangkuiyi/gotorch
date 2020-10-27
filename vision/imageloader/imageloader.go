package imageloader

import (
	"fmt"
	"image"
	"io"
	"math/rand"
	"path/filepath"
	"runtime"
	"sync"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
	"gocv.io/x/gocv"
)

type sample struct {
	data  gocv.Mat
	label int64
}

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
	sampleChan chan sample
	mbChan     chan miniBatch
	errChan    chan error
	trans1     *transforms.ComposeTransformer // transforms before `ToTensor`
	trans2     *transforms.ComposeTransformer // transforms after and include `ToTensor`
	mbSize     int
	miniBatch  miniBatch
	err        error
	pinMemory  bool
	colorSpace string
	// shuffle buffer configuration
	bufSize     int
	shuffleChan chan sample
	seed        int64
}

var (
	readSamplesCh          = make(chan func(), 1)
	samplesToMinibatchesCh = make(chan func(), 1)
	activeInstanceCounter  = 0 // Track the number of instances of the type `ImageLoader`
	counterMu              = &sync.Mutex{}
	threadGroupWaterline   = 0 // The number of working thread groups should be as large as the max value of `activeInstanceCounter`
)

// New returns an ImageLoader
func New(fn string, vocab map[string]int, trans *transforms.ComposeTransformer,
	mbSize, bufSize int, seed int64, pinMemory bool, colorSpace string) (*ImageLoader, error) {
	if mbSize <= 0 {
		panic("mbSize(batch size) should be greater than 0")
	}
	r, e := tgz.OpenFile(fn)
	if e != nil {
		return nil, e
	}
	trans1, trans2 := splitComposeByToTensor(trans)
	m := &ImageLoader{
		r:           r,
		vocab:       vocab,
		sampleChan:  make(chan sample, mbSize*4),
		mbChan:      make(chan miniBatch, 4),
		shuffleChan: make(chan sample, mbSize*4),
		errChan:     make(chan error, 1),
		trans1:      trans1,
		trans2:      trans2,
		mbSize:      mbSize,
		pinMemory:   pinMemory,
		colorSpace:  colorSpace,
		bufSize:     bufSize,
		seed:        seed,
	}
	go m.shuffleSamples()

	// `readSamples` and `samplesToMinibatches` calls `gocv`, we make them only run
	// in background OS threads. Otherwise, if we run them in goroutines like
	// `go m.readSamples()` and `go samplesToMinibatches`, the two goroutines will
	// create many OS threads without `runtime.LockOSThread` because of the Cgo
	// mechanism, or cause memory leak with a `runtime.LockOSThread` call. See the
	// comment of `func newWorkingThreadGroup` for the thread explosion issue and
	// https://github.com/opencv/opencv/issues/9745 for the memory leak issue.
	readSamplesCh <- func() { m.readSamples() }
	samplesToMinibatchesCh <- func() { m.samplesToMinibatches() }
	counterMu.Lock()
	defer counterMu.Unlock()
	activeInstanceCounter++
	if activeInstanceCounter > threadGroupWaterline {
		newWorkingThreadGroup()
		threadGroupWaterline = activeInstanceCounter
	}
	runtime.SetFinalizer(m, func(m *ImageLoader) {
		counterMu.Lock()
		activeInstanceCounter--
		counterMu.Unlock()
	})
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
		p.miniBatch = miniBatch
		return true
	}
	torch.FinishGC()
	return false
}

func (p *ImageLoader) readSamples() {
	defer func() {
		close(p.sampleChan)
	}()

	for {
		hdr, err := p.r.Next()
		if err != nil {
			if err != io.EOF {
				p.errChan <- err
			}
			break
		}
		if !hdr.FileInfo().Mode().IsRegular() {
			continue
		}
		classStr := filepath.Base(filepath.Dir(hdr.Name))
		label := p.vocab[classStr]
		buffer := make([]byte, hdr.Size)
		io.ReadFull(p.r, buffer)
		m, err := decodeImage(buffer, p.colorSpace)
		if err != nil {
			p.errChan <- err
			break
		}
		if m.Empty() {
			panic("read invalid image content!")
		}
		p.sampleChan <- sample{p.trans1.Run(m).(gocv.Mat), int64(label)}
	}
}

func (p *ImageLoader) samplesToMinibatches() {
	inputs := []gocv.Mat{}
	labels := []int64{}
	defer func() {
		close(p.mbChan)
		close(p.errChan)
	}()
	for {
		sample, ok := <-p.shuffleChan
		if !ok {
			p.errChan <- io.EOF
			break
		}
		inputs = append(inputs, sample.data)
		labels = append(labels, sample.label)
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

func (p *ImageLoader) shuffleSamples() {
	buffer := []sample{}
	rand.Seed(p.seed)
	defer close(p.shuffleChan)
	for i := 0; i < p.bufSize; i++ {
		sample, ok := <-p.sampleChan
		if !ok {
			break
		}
		buffer = append(buffer, sample)
	}
	for {
		sample, ok := <-p.sampleChan
		if !ok {
			break
		}
		randIdx := rand.Intn(len(buffer))
		p.shuffleChan <- buffer[randIdx]
		buffer[randIdx] = sample
	}
	rand.Shuffle(len(buffer), func(i, j int) { buffer[i], buffer[j] = buffer[j], buffer[i] })
	for _, sample := range buffer {
		p.shuffleChan <- sample
	}
}

func (p *ImageLoader) collateMiniBatch(inputs []gocv.Mat, labels []int64) miniBatch {
	w := inputs[0].Cols()
	h := inputs[0].Rows()
	blob := gocv.NewMat()
	defer func() {
		// `gocv.Mat`s must be released manually
		blob.Close()
		for _, i := range inputs {
			i.Close()
		}
	}()
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
	return p.miniBatch.data, p.miniBatch.label
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

func decodeImage(buffer []byte, colorSpace string) (gocv.Mat, error) {
	var m gocv.Mat
	var e error
	if colorSpace == RGB {
		m, e = gocv.IMDecode(buffer, gocv.IMReadColor)
	} else if colorSpace == GRAY {
		m, e = gocv.IMDecode(buffer, gocv.IMReadGrayScale)
	} else {
		return m, fmt.Errorf("Cannot read image with color space %v", colorSpace)
	}
	if colorSpace == RGB {
		gocv.CvtColor(m, &m, gocv.ColorBGRToRGB)
	}
	return m, e
}

func newWorkingThreadGroup() {
	// We use these background threads to call `gocv`. This is because `gocv` makes
	// `Cgo` calls extensively, if we call `gocv` directly in goroutines(each epoch
	// creates a new goroutine), the `Cgo` calls will cause Go runtime to create too
	// many threads.
	go func() {
		runtime.LockOSThread()
		for f := range readSamplesCh {
			f()
		}
	}()
	go func() {
		runtime.LockOSThread()
		for f := range samplesToMinibatchesCh {
			f()
		}
	}()
}
