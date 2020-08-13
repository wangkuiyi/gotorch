package nn

import (
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

// BatchNorm2d torch.nn.BatchNorm2d
// TODO(qijun): training flag is always true
type BatchNorm2d struct {
	Module
	NumFeatures       int64
	Eps               float64
	Momentum          float64
	Affine            bool
	TrackRunningStats bool
	Weight            torch.Tensor
	Bias              torch.Tensor
	RunningMean       torch.Tensor `gotorch:"buffer"`
	RunningVar        torch.Tensor `gotorch:"buffer"`
}

// NewBatchNorm2d creates a `BatchNorm2d` instance
func NewBatchNorm2d(numFeatures int64, eps, momentum float64,
	affine, trackRunningStats bool) *BatchNorm2d {
	b := &BatchNorm2d{
		Module:            Module{isTraining: true},
		NumFeatures:       numFeatures,
		Eps:               eps,
		Momentum:          momentum,
		Affine:            affine,
		TrackRunningStats: trackRunningStats,
	}
	if b.Affine {
		b.Weight = torch.Empty([]int64{numFeatures}, true)
		b.Bias = torch.Empty([]int64{numFeatures}, true)
	}
	if b.TrackRunningStats {
		b.RunningMean = torch.Empty([]int64{numFeatures}, false)
		b.RunningVar = torch.Empty([]int64{numFeatures}, false)
	}
	b.resetParameters()
	b.Init(b)
	return b
}

func (b *BatchNorm2d) resetRunningStats() {
	if b.TrackRunningStats {
		initializer.Zeros(&b.RunningMean)
		initializer.Ones(&b.RunningVar)
	}
}

func (b *BatchNorm2d) resetParameters() {
	b.resetRunningStats()
	if b.Affine {
		initializer.Ones(&b.Weight)
		initializer.Zeros(&b.Bias)
	}
}

// Forward method
func (b *BatchNorm2d) Forward(x torch.Tensor) torch.Tensor {
	bnTraining := (b.RunningMean.T == nil) && (b.RunningVar.T == nil)
	if b.isTraining {
		bnTraining = true
	}
	var fmean, fvar torch.Tensor
	if !b.isTraining || b.TrackRunningStats {
		fmean = b.RunningMean
		fvar = b.RunningVar
	}
	return functional.BatchNorm(x, fmean, fvar, b.Weight, b.Bias, bnTraining,
		b.Momentum, b.Eps)
}
