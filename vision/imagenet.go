package vision

import (
	"os"
	"path/filepath"
	"sort"

	torch "github.com/wangkuiyi/gotorch"
)

type imageSample struct {
	path     string
	classIdx int
}

// ImageNet provides a convenient interface to reader
// batch from ImageNet dataset.
// Usage:
//
// transforms:= []Transform{ToTensor{}, RandomResizedCrop{}})
// imageNet := NewImageNet("~/.cache/imagenet/train", 32, transforms)
// for range imageNet.Scan() {
//	img, target := imageNet.Batch()
// }
type ImageNet struct {
	batchSize  int64
	samples    []imageSample
	transforms []Transform
	// iterator status
	iterIdx      int
	batchIdxList []int
}

// Transform interface
type Transform interface {
	Do(torch.Tensor) torch.Tensor
}

// NewImageNet returns ImageNet dataloader
func NewImageNet(root string, batchSize int64, transforms []Transform) *ImageNet {
	imageNet := &ImageNet{
		batchSize:  batchSize,
		transforms: transforms,
		iterIdx:    0,
	}
	imageNet.samples = makeSamples(root)
	return imageNet
}

// Batch returns data and target Tensor
func (p *ImageNet) Batch() (torch.Tensor, torch.Tensor) {
	dataArray := []torch.Tensor{}
	targetArray := []torch.Tensor{}
	for range p.batchIdxList {
		// TODO(yancey1989): implement the following transformation functions
		// `torch.ToTensor` transforms Image object to torch.Tensor
		// `torch.RandomResizedCrop`
		// `torch.RandomHorizontalFlip`
		// `torch.Normalize`
		dataArray = append(dataArray, torch.RandN([]int64{3, 2}, false))
		targetArray = append(targetArray, torch.RandN([]int64{3}, false))
	}
	// TODO(yancey1989): implement stack to stack tensor array to onetensor
	// dataTensor := torch.Stack(dataArray)
	// targetTensor := torch.Stack(targetArray)
	return torch.RandN([]int64{p.batchSize, 3, 2}, false), torch.RandN([]int64{p.batchSize, 1}, false)
}

// Scan return false if no more data
func (p *ImageNet) Scan() bool {
	p.batchIdxList = []int{}
	if p.isEOF() {
		return false
	}
	for i := int64(0); i < p.batchSize; i++ {
		p.batchIdxList = append(p.batchIdxList, p.iterIdx)
		p.iterIdx++
		if p.isEOF() {
			break
		}
	}
	return true
}

func (p *ImageNet) isEOF() bool {
	if p.iterIdx >= len(p.samples) {
		return true
	}
	return false
}

func makeSamples(root string) []imageSample {
	samples := []imageSample{}
	file, e := os.Open(root)
	if e != nil {
		panic(e)
	}
	classes, e := file.Readdirnames(0)
	if e != nil {
		panic(e)
	}
	sort.Strings(classes)

	for idx, targetClass := range classes {
		targetDir := filepath.Join(root, targetClass)
		err := filepath.Walk(targetDir, func(path string, info os.FileInfo, err error) error {
			samples = append(samples, imageSample{
				path:     path,
				classIdx: idx,
			})
			return nil
		})
		if err != nil {
			panic(err)
		}
	}
	return samples
}
