package datasets

import (
	"image"
	"image/color"
	"io/ioutil"
	"math/rand"
	"os"
	"path"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

// The dataset layout description locates at https://www.cs.toronto.edu/~kriz/cifar.html
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
			rgba.Set(ix, iy, color.RGBA{uint8(data[ix*y+iy]),
				uint8(data[length+ix*y+iy]),
				uint8(data[2*length+ix*y+iy]),
				255})
		}
	}
	return rgba
}

func randomSampler(size int) map[int]int {
	res := make(map[int]int)
	for i, v := range rand.Perm(size) {
		res[i] = v
	}
	return res
}

// CIFAR10Loader struct
type CIFAR10Loader struct {
	batchSize int
	offset    int
	root      string
	train     bool
	shuffle   bool
	data      []image.Image
	target    []int
	sampler   map[int]int
	trans     *transforms.ComposeTransformer
}

// CIFAR10 creates a CIFAR10Loader instance
func CIFAR10(root string, train bool, shuffle bool, batchSize int, trans *transforms.ComposeTransformer) (*CIFAR10Loader, error) {
	c := &CIFAR10Loader{
		batchSize: batchSize,
		root:      root,
		train:     train,
		shuffle:   shuffle,
		trans:     trans,
		data:      make([]image.Image, 0),
		target:    make([]int, 0),
	}
	imgSize := 3 * 32 * 32
	downloadList := cifar10TestList
	if c.train {
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
			c.data = append(c.data, src)
			c.target = append(c.target, int(target))
		}
		file.Close()
	}
	if c.shuffle {
		c.sampler = randomSampler(len(c.data))
	} else {
		c.sampler = make(map[int]int)
		for i := 0; i < len(c.data); i++ {
			c.sampler[i] = i
		}
	}
	return c, nil
}

// Batch returns a minibatch with data and label Tensor
func (c *CIFAR10Loader) Batch() (torch.Tensor, torch.Tensor) {
	dataSlice := []torch.Tensor{}
	labelSlice := []torch.Tensor{}
	for i := c.offset; (i < c.offset+c.batchSize) && i < len(c.data); i++ {
		j := c.sampler[i]
		data := c.trans.Run(c.data[j])
		label := transforms.ToTensor().Run(c.target[j])
		dataSlice = append(dataSlice, data.(torch.Tensor))
		labelSlice = append(labelSlice, label)
	}
	c.offset = c.offset + c.batchSize
	return torch.Stack(dataSlice, 0), torch.Stack(labelSlice, 0)
}

// Scan scans the batch from Loader
func (c *CIFAR10Loader) Scan() bool {
	torch.GC()
	if c.offset >= len(c.data) {
		return false
	}
	return true
}

// Reset the offset
func (c *CIFAR10Loader) Reset() {
	c.offset = 0
}
