package nn

import (
	"math"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

// Conv2d applies convolution over a 2D input.
type Conv2d struct {
	Module
	InChannels  int64
	OutChannels int64
	KernelSize  int64
	Stride      int64
	Padding     int64
	Dilation    int64
	Groups      int64
	PaddingMode string
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// NewConv2d creates a `Conv2d` instance
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
func NewConv2d(inChannels, outChannels, kernelSize, stride, padding, dilation,
	groups int64, bias bool, paddingMode string) *Conv2d {
	c := &Conv2d{
		Module:      Module{isTraining: true},
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Dilation:    dilation,
		Groups:      groups,
		PaddingMode: "zeros",
	}
	c.Weight = torch.Empty([]int64{outChannels, inChannels / groups, kernelSize,
		kernelSize}, true)
	if bias {
		c.Bias = torch.Empty([]int64{outChannels}, true)
	}
	c.ResetParameters()
	return c
}

func (c *conv2d) ResetParameters() {
	initializer.KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if c.Bias.T != nil {
		fanIn, _ := initializer.CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&c.Bias, -bound, bound)
	}
	c.Init(c)
	return c
}

// Forward method
func (c *Conv2d) Forward(x torch.Tensor) torch.Tensor {
	return functional.Conv2d(x, c.Weight, c.Bias, []int64{c.Stride, c.Stride},
		[]int64{c.Padding, c.Padding}, []int64{c.Dilation, c.Dilation}, c.Groups)
}
