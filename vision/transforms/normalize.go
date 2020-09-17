package transforms

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

// NormalizeTransformer corresponds to torchvision.transforms.html#Normalize. It
// implements Go interface gotorch/data.Transform.
type NormalizeTransformer struct {
	mean, stddev torch.Tensor
}

// Normalize returns normalize transformer
func Normalize(mean []float32, stddev []float32) *NormalizeTransformer {
	var meanT torch.Tensor
	var stddevT torch.Tensor
	if len(mean) == 1 {
		meanT = torch.NewTensor([][][]float32{{{mean[0]}}})
	} else if len(mean) == 3 {
		meanT = torch.NewTensor([][][]float32{{{mean[0]}}, {{mean[1]}}, {{mean[2]}}})
	} else {
		panic(fmt.Sprintf("len(Mean) should be 1 or 3."))
	}
	if len(stddev) == 1 {
		stddevT = torch.NewTensor([][][]float32{{{stddev[0]}}})
	} else if len(stddev) == 3 {
		stddevT = torch.NewTensor([][][]float32{{{stddev[0]}}, {{stddev[1]}}, {{stddev[2]}}})
	} else {
		panic(fmt.Sprintf("len(Stddev) should be 1 or 3."))
	}
	return &NormalizeTransformer{meanT, stddevT}
}

// Run normalize the input (Tensor) of size (C, H, W) using the stats value mean, stddev.
func (t *NormalizeTransformer) Run(input torch.Tensor) torch.Tensor {
	// TODO(typhoonzero): check if stddevT equals 0
	x := input.Sub(t.mean, 1.0)
	x = x.Div(t.stddev)
	return x
}
