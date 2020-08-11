package nn

import (
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

// TODO(qijun): training flag is always true
type batchNorm2d struct {
	NumFeatures       int64
	Eps               float64
	Momentum          float64
	Affine            bool
	TrackRunningStats bool
	Weight            torch.Tensor
	Bias              torch.Tensor
	RunningMean       torch.Tensor `gotorch:"buffer"`
	RunningVar        torch.Tensor `gotorch:"buffer"`
	Training          bool
}

// BatchNorm2d torch.nn.BatchNorm2d
func BatchNorm2d(numFeatures int64, eps, momentum float64,
	affine, trackRunningStats bool) Module {
	b := &batchNorm2d{
		NumFeatures:       numFeatures,
		Eps:               eps,
		Momentum:          momentum,
		Affine:            affine,
		TrackRunningStats: trackRunningStats,
		Training:          true,
	}
	if b.Affine {
		b.Weight = torch.Empty([]int64{numFeatures}, true)
		b.Bias = torch.Empty([]int64{numFeatures}, true)
	}
	if b.TrackRunningStats {
		b.RunningMean = torch.Empty([]int64{numFeatures}, false)
		b.RunningVar = torch.Empty([]int64{numFeatures}, false)
	}
	b.ResetParameters()
	return b
}

func (b *batchNorm2d) ResetRunningStats() {
	if b.TrackRunningStats {
		initializer.Zeros(&b.RunningMean)
		initializer.Ones(&b.RunningVar)
	}
}

func (b *batchNorm2d) ResetParameters() {
	b.ResetRunningStats()
	if b.Affine {
		initializer.Ones(&b.Weight)
		initializer.Zeros(&b.Bias)
	}
}

func (b *batchNorm2d) Forward(x torch.Tensor) torch.Tensor {
	bnTraining := (b.RunningMean.T == nil) && (b.RunningVar.T == nil)
	if b.Training {
		bnTraining = true
	}
	var fmean, fvar torch.Tensor
	if !b.Training || b.TrackRunningStats {
		fmean = b.RunningMean
		fvar = b.RunningVar
	}
	return functional.BatchNorm(x, fmean, fvar, b.Weight, b.Bias, bnTraining,
		b.Momentum, b.Eps)
}
