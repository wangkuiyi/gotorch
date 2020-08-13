package nn

import (
	"math"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	initializer "github.com/wangkuiyi/gotorch/nn/initializer"
)

// LinearModule applies a linear transformation with optional bias.
type LinearModule struct {
	Module
	InFeatures  int64
	OutFeatures int64
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// Linear creates a `Linear` instance
func Linear(in, out int64, bias bool) *LinearModule {
	l := &LinearModule{
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
func (l *LinearModule) Forward(x torch.Tensor) torch.Tensor {
	return F.Linear(x, l.Weight, l.Bias)
}

func (l *LinearModule) resetParameters() {
	initializer.KaimingUniform(&l.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if l.Bias.T != nil {
		fanIn, _ := initializer.CalculateFanInAndFanOut(l.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&l.Bias, -bound, bound)
	}
}
