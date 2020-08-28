package transforms

import (
	"fmt"

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
func (t *NormalizeTransformer) Run(input torch.Tensor) torch.Tensor {
	dtype := input.Dtype()
	var meanT torch.Tensor
	var stddevT torch.Tensor
	if len(t.Mean) == 1 {
		meanT = torch.NewTensor([][][]float64{{{t.Mean[0]}}})
	} else if len(t.Mean) == 3 {
		meanT = torch.NewTensor([][][]float64{{{t.Mean[0]}}, {{t.Mean[1]}}, {{t.Mean[2]}}})
	} else {
		panic(fmt.Sprintf("len(Mean) should be 1 or 3."))
	}
	if len(t.Stddev) == 1 {
		stddevT = torch.NewTensor([][][]float64{{{t.Stddev[0]}}})
	} else if len(t.Stddev) == 3 {
		stddevT = torch.NewTensor([][][]float64{{{t.Stddev[0]}}, {{t.Stddev[1]}}, {{t.Stddev[2]}}})
	} else {
		panic(fmt.Sprintf("len(Stddev) should be 1 or 3."))
	}
	// TODO(typhoonzero): check if stddevT equals 0
	x := input.Sub(meanT, 1.0)
	x = x.Div(stddevT).CastTo(dtype)
	return x
}
