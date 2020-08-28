package transforms

import torch "github.com/wangkuiyi/gotorch"

// PermuteTransformer permute the input tensor to desired
type PermuteTransformer struct {
	dims []int64
}

// Permute permte the tensor shape and return.
func Permute(dims []int64) *PermuteTransformer {
	return &PermuteTransformer{dims}
}

// Run runs the permute transform.
func (t *PermuteTransformer) Run(input interface{}) interface{} {
	inputTensor, ok := input.(torch.Tensor)
	if !ok {
		panic("NormalizeTransformer accepts Tensor input only.")
	}
	return inputTensor.Permute(t.dims)
}
