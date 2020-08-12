package nn

import (
	"math"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

type conv2d struct {
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

// Conv2d does conv2d computaion. torch.nn.functional.conv2d
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
func Conv2d(inChannels, outChannels, kernelSize, stride, padding, dilation,
	groups int64, bias bool, paddingMode string) Module {
	c := &conv2d{
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
	initializer.KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if bias {
		c.Bias = torch.Empty([]int64{outChannels}, true)
		fanIn, _ := initializer.CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&c.Bias, -bound, bound)
	}
	return c
}

// Forward method
func (c *conv2d) Forward(x torch.Tensor) torch.Tensor {
	return functional.Conv2d(x, c.Weight, c.Bias, []int64{c.Stride, c.Stride},
		[]int64{c.Padding, c.Padding}, []int64{c.Dilation, c.Dilation}, c.Groups)
}

type convTranspose2d struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	OutPadding  int
	Groups      int
	Dilation    int
	PaddingMode string
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// ConvTranspose2d torch.nn.functional.conv_transpose2d
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
func ConvTranspose2d(inChannels, outChannels, kernelSize, stride, padding,
	outPadding, groups, dilation int, bias bool, paddingMode string) Module {
	c := &convTranspose2d{
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
	c.Weight = torch.Empty([]int{inChannels, outChannels / groups, kernelSize,
		kernelSize}, true)
	initializer.KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if bias {
		c.Bias = torch.Empty([]int{outChannels}, true)
		fanIn, _ := initializer.CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		initializer.Uniform(&c.Bias, -bound, bound)
	}
	return c
}

// Forward method
func (c *convTranspose2d) Forward(x torch.Tensor) torch.Tensor {
	return functional.ConvTranspose2d(x, c.Weight, c.Bias,
		[]int{c.Stride, c.Stride}, []int{c.Padding, c.Padding},
		[]int{c.OutPadding, c.OutPadding}, c.Groups, []int{c.Dilation, c.Dilation})
}
