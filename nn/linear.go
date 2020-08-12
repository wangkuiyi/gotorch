package nn

import torch "github.com/wangkuiyi/gotorch"

// Linear applies a linear transformation with optional bias.
type Linear struct {
	Module
	InFeatures  int64
	OutFeatures int64
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// NewLinear creates a `Linear` instance
func NewLinear(in, out int64, bias bool) *Linear {
	l := &Linear{
		Module:      Module{isTraining: true},
		InFeatures:  in,
		OutFeatures: out,
	}
	l.Weight = torch.RandN([]int64{in, out}, true)
	if bias {
		l.Bias = torch.RandN([]int64{out, 1}, true)
	}
	l.Init(l)
	return l
}

// Forward does a linear transformation to the `input` tensor.
func (l *Linear) Forward(x torch.Tensor) torch.Tensor {
	return torch.MM(x, l.Weight)
}
