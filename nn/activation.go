package nn

import (
	torch "github.com/wangkuiyi/gotorch"
)

// LeakyReluModule torch.nn.LeakyRelu
// TODO(qijun): training flag is always true
type LeakyReluModule struct {
	Module
	NegativeSlope float64
	Inplace       bool
}

// LeakyRelu creates a `LeakyReluModule` instance
func LeakyRelu(negativeSlope float64, inplace bool) *LeakyReluModule {
	return &LeakyReluModule{
		Module:        Module{isTraining: true},
		NegativeSlope: negativeSlope,
		Inplace:       inplace,
	}
}

// Forward method
func (l *LeakyReluModule) Forward(x torch.Tensor) torch.Tensor {
	return torch.LeakyRelu(x, l.NegativeSlope)
}
