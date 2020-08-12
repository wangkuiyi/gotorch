package nn

import torch "github.com/wangkuiyi/gotorch"

type linear struct {
	InFeatures  int64
	OutFeatures int64
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// Linear creates a linear instance
func Linear(in, out int64, bias bool) Module {
	l := &linear{
		InFeatures:  in,
		OutFeatures: out,
	}
	l.Weight = torch.RandN([]int64{in, out}, true)
	if bias {
		l.Bias = torch.RandN([]int64{out, 1}, true)
	}
	return l
}

// Forward method
func (l *linear) Forward(x torch.Tensor) torch.Tensor {
	return torch.MM(x, l.Weight)
}
