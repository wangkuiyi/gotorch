package nn

import (
	"math"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	initializer "github.com/wangkuiyi/gotorch/nn/initializer"
)

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
	l.Weight = torch.Empty([]int64{out, in}, true)
	if bias {
		l.Bias = torch.Empty([]int64{out, 1}, true)
	}
	l.Init(l)
	l.resetParameters()
	return l
}

// Forward does a linear transformation to the `input` tensor.
func (l *Linear) Forward(x torch.Tensor) torch.Tensor {
	return F.Linear(x, l.Weight, l.Bias)
}

func (l *Linear) resetParameters() {
	initializer.KaimingUniform(&l.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if l.Bias.T != nil {
		fanIn, _ := initializer.CalculateFanInAndFanOut(l.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&l.Bias, -bound, bound)
	}
}
