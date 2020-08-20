package data

import (
	torch "github.com/wangkuiyi/gotorch"
)

// Dataset interface represents a map from keys to
// samples. All Datasets should implements `GetItem`
// to fetch a sample by the given index. `Len` to
// fetch the size of the Dataset.
type Dataset interface {
	GetItem(index int) (torch.Tensor, torch.Tensor, error)
	Len() int
}

// Transform interface represents a transformations on data.
// All Transforms should implement `Do` to return a transformation
// data.
type Transform interface {
	Do(torch.Tensor) torch.Tensor
}
