package data

import (
	"testing"

	torch "github.com/wangkuiyi/gotorch"
)

type EmptyTensorTransform struct {
	shape []int64
}

type Random struct {
	dataShape   []int64
	targetShape []int64
	transforms  []Transform
}

func NewRandom(dataShape []int64, targetShape []int64, transforms []Transform) Dataset {
	return &Random{
		dataShape:   dataShape,
		targetShape: targetShape,
		transforms:  transforms,
	}
}

func (rand *Random) GetItem(index int) (torch.Tensor, torch.Tensor, error) {
	data, target := torch.RandN(rand.dataShape, false), torch.RandN(rand.targetShape, false)
	for _, t := range rand.transforms {
		data = t.Do(data)
	}
	return data, target, nil
}

func (rand *Random) Len() int {
	return 10
}

func (t EmptyTensorTransform) Do(torch.Tensor) torch.Tensor {
	return torch.Empty(t.shape, false)
}

func TestLoader(t *testing.T) {
	emptyTensorTransform := &EmptyTensorTransform{[]int64{3, 2}}
	randomDataset := NewRandom([]int64{3, 2}, []int64{3, 2}, []Transform{emptyTensorTransform})
	loader := NewLoader(randomDataset, 32)
	for loader.Scan() {
		loader.Batch()
	}
}
