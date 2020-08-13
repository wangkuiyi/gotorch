package nn

import (
	"math"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

// Conv2dModule applies convolution over a 2D input.
type Conv2dModule struct {
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

// Conv2d creates a `Conv2dModule` instance
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
func Conv2d(inChannels, outChannels, kernelSize, stride, padding, dilation,
	groups int64, bias bool, paddingMode string) *Conv2dModule {
	c := &Conv2dModule{
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
	c.Init(c)
	c.resetParameters()
	return c
}

func (c *Conv2dModule) resetParameters() {
	initializer.KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if c.Bias.T != nil {
		fanIn, _ := initializer.CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&c.Bias, -bound, bound)
	}
}

// Forward method
func (c *Conv2dModule) Forward(x torch.Tensor) torch.Tensor {
	return functional.Conv2d(x, c.Weight, c.Bias, []int64{c.Stride, c.Stride},
		[]int64{c.Padding, c.Padding}, []int64{c.Dilation, c.Dilation}, c.Groups)
}

// ConvTranspose2dModule corresponds to torch.nn.ConvTranspose2d
type ConvTranspose2dModule struct {
	Module
	InChannels  int64
	OutChannels int64
	KernelSize  int64
	Stride      int64
	Padding     int64
	OutPadding  int64
	Groups      int64
	Dilation    int64
	PaddingMode string
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// ConvTranspose2d torch.nn.conv_transpose2d
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
// not support output_size when forwarding
func ConvTranspose2d(inChannels, outChannels, kernelSize, stride, padding,
	outPadding, groups int64, bias bool, dilation int64, paddingMode string) *ConvTranspose2dModule {
	c := &ConvTranspose2dModule{
		Module:      Module{isTraining: true},
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		OutPadding:  outPadding,
		Groups:      groups,
		Dilation:    dilation,
		PaddingMode: "zeros",
	}
	c.Weight = torch.Empty([]int64{inChannels, outChannels / groups, kernelSize,
		kernelSize}, true)
	if bias {
		c.Bias = torch.Empty([]int64{outChannels}, true)
	}
	c.Init(c)
	c.resetParameters()
	return c
}

func (c *ConvTranspose2dModule) resetParameters() {
	initializer.KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if c.Bias.T != nil {
		fanIn, _ := initializer.CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&c.Bias, -bound, bound)
	}
}

// Forward method
func (c *ConvTranspose2dModule) Forward(x torch.Tensor) torch.Tensor {
	return functional.ConvTranspose2d(x, c.Weight, c.Bias,
		[]int64{c.Stride, c.Stride}, []int64{c.Padding, c.Padding},
		[]int64{c.OutPadding, c.OutPadding}, c.Groups, []int64{c.Dilation, c.Dilation})
}
