package data

import (
	torch "github.com/wangkuiyi/gotorch"
)

// Example contains data and target
type Example struct {
	data, target torch.Tensor
	hasGCed      bool
}

// NewExample creates an example from `data` and `target`
func NewExample(data, target torch.Tensor) *Example {
	return &Example{data, target, false}
}

// Data of the example
func (e *Example) Data() torch.Tensor {
	if !e.hasGCed {
		torch.GC()
		e.hasGCed = true
	}
	torch.SetTensorFinalizer(e.data.T)
	return e.data
}

// Target of the example
func (e *Example) Target() torch.Tensor {
	if !e.hasGCed {
		torch.GC()
		e.hasGCed = true
	}
	torch.SetTensorFinalizer(e.target.T)
	return e.target
}

// Dataset is the interface of datasets
type Dataset interface {
	Get() *Example
	Reset()
}

// Loader is a generator utility function for range over a `dataset`
// Usage:
//     for batch := range Loader(myDataset) {
//         ...
//     }
func Loader(dataset Dataset) chan Example {
	c := make(chan Example, 0)
	dataset.Reset()
	go func() {
		defer close(c)
		for {
			e := dataset.Get()
			if e == nil {
				break
			}
			c <- *e
		}
	}()
	return c
}
