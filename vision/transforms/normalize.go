package transforms

import (
	torch "github.com/wangkuiyi/gotorch"
)

// NormalizeTransformer corresponds to torchvision.transforms.html#Normalize. It
// implements Go interface gotorch/data.Transform.
type NormalizeTransformer struct {
	Mean, Stddev []float64
}

// Normalize returns normalize transformer
func Normalize(mean []float64, stddev []float64) *NormalizeTransformer {
	return &NormalizeTransformer{mean, stddev}
}

// Run normalize the input (Tensor) of size (C, H, W) using the stats value mean, stddev.
func (t *NormalizeTransformer) Run(input interface{}) interface{} {
	inputTensor, ok := input.(torch.Tensor)
	if !ok {
		panic("NormalizeTransformer accepts Tensor input only.")
	}
	meanT := torch.NewTensor(t.Mean).View([]int64{-1, 1, 1})
	// TODO(typhoonzero): check if stddevT equals 0
	stddevT := torch.NewTensor(t.Stddev).View([]int64{-1, 1, 1})
	inputTensor.SubI(meanT, 1.0)
	inputTensor.DivI(stddevT)
	return inputTensor
}
