package data

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

// Batch struct contains Data and Target Tensor
type Batch struct {
	Data   torch.Tensor
	Target torch.Tensor
}

// Loader provides a convenient interface to read batch
// from a dataset.
type Loader struct {
	dataset   Dataset
	batchSize int64
	// sequence sampler status
	// TODO(yancey1989) we need Sampler interface
	// if supporting user custom Sampler
	idxList []int // collect the index samples into one tensor
	idx     int   // current sample index in dataset
}

// Scan return false if no more data
func (loader *Loader) Scan() bool {
	// stop scanner on EOF
	if loader.idx > loader.dataset.Len() {
		return false
	}
	loader.idxList = []int{}

	for i := int64(0); i < loader.batchSize; i++ {
		loader.idx++
		if loader.idx > loader.dataset.Len() {
			break
		}
		loader.idxList = append(loader.idxList, loader.idx)
	}
	return true
}

// Batch returns the current Batch
func (loader *Loader) Batch() *Batch {
	dataArray := []torch.Tensor{}
	targetArray := []torch.Tensor{}
	for _, idx := range loader.idxList {
		data, target, e := loader.dataset.GetItem(idx)
		if e != nil {
			panic(fmt.Sprintf("get item failed on index: %d, %e", idx, e))
		}
		dataArray = append(dataArray, data)
		targetArray = append(targetArray, target)
	}
	// TODO(yancey1989): call `torch.Stack` to stack tensor array into one tensor
	return &Batch{
		Data:   torch.RandN([]int64{loader.batchSize, 3, 2}, false),
		Target: torch.RandN([]int64{loader.batchSize, 1}, false),
	}
}

// NewLoader returns a Loader
func NewLoader(dataset Dataset, batchSize int64) *Loader {
	return &Loader{
		dataset:   dataset,
		batchSize: batchSize,
		idx:       -1,
		idxList:   []int{},
	}
}
