package datasets

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"os"
	"path"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

var cifar10Url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
var cifar10BaseFolder = "cifar-10-batches-bin"
var cifar10FileName = "cifar-10-binary.tar.gz"
var cifar10TrainList = []string{"data_batch_1.bin",
	"data_batch_2.bin", "data_batch_3.bin",
	"data_batch_4.bin", "data_batch_5.bin"}

var cifar10TestList = []string{"test_batch.bin"}

var meta = "batches.meta.txt"

func rgbBytesToImage(data []byte, x, y int) *image.RGBA {
	rgba := image.NewRGBA(image.Rect(0, 0, x, y))
	length := x * y
	for ix := 0; ix < x; ix++ {
		for iy := 0; iy < y; iy++ {
			rgba.Set(ix, iy, color.NRGBA{uint8(data[ix*y+iy]),
				uint8(data[length+ix*y+iy]),
				uint8(data[2*length+ix*y+iy]),
				255})
		}
	}
	return rgba
}

// CIFAR10Loader struct
type CIFAR10Loader struct {
	batchSize int64
	offset    int64
	root      string
	train     bool
	samples   []*Sample
	trans     *transforms.ComposeTransformer
}

// CIFAR10 creates a CIFAR10Loader instance
func CIFAR10(root string, train bool, batchSize int64, trans *transforms.ComposeTransformer) (*CIFAR10Loader, error) {
	c := &CIFAR10Loader{
		batchSize: batchSize,
		root:      root,
		train:     train,
		trans:     trans,
		samples:   make([]*Sample, 0),
	}
	imgSize := 3 * 32 * 32
	downloadList := cifar10TestList
	if train {
		downloadList = cifar10TrainList
	}
	for _, fileName := range downloadList {
		filePath := path.Join(c.root, cifar10BaseFolder, fileName)
		file, err := os.Open(filePath)
		if err != nil {
			return nil, err
		}
		b, _ := ioutil.ReadAll(file)
		for i := 0; i < 10000; i++ {
			target := uint8(b[(imgSize+1)*i])
			data := b[(imgSize+1)*i+1 : (imgSize+1)*(i+1)]
			src := rgbBytesToImage(data, 32, 32)
			c.samples = append(c.samples, &Sample{src, int(target)})
		}
		file.Close()
	}
	return c, nil
}

// Batch returns a minibatch with data and label Tensor
func (c *CIFAR10Loader) Batch() (torch.Tensor, torch.Tensor) {
	dataArray := []torch.Tensor{}
	labelArray := []torch.Tensor{}
	for i := c.offset; (i < c.offset+c.batchSize) && i < int64(len(c.samples)); i++ {
		fmt.Println(i)
		data := c.trans.Run(c.samples[i].image)
		label := transforms.ToTensor().Run(c.samples[i].target)
		dataArray = append(dataArray, data.(torch.Tensor))
		labelArray = append(labelArray, label)
	}
	c.offset = c.offset + c.batchSize
	return torch.Stack(dataArray, 0), torch.Stack(labelArray, 0)
}

// Scan scans the batch from Loader
func (c *CIFAR10Loader) Scan() bool {
	torch.GC()
	if c.offset >= int64(len(c.samples)) {
		return false
	}
	return true
}

// Reset the offset
func (c *CIFAR10Loader) Reset() {
	c.offset = 0
}
