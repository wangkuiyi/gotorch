package module

import (
	"math"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn/functional"
)

type conv2d struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Dilation    int
	Groups      int
	PaddingMode string
	Weight      torch.Tensor
	Bias        torch.Tensor
}

// Conv2d does conv2d computaion. torch.conv2d
// TODO(qijun): only support zero padding mode
// only support symmetry kernel/stride/padding/dilation
func Conv2d(inChannels, outChannels, kernelSize, stride, padding, dilation,
	groups int, bias bool, paddingMode string) Module {
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
	c.Weight = torch.Empty([]int{outChannels, inChannels / groups, kernelSize,
		kernelSize}, true)
	torch.KaimingUniform(&c.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
	if bias {
		c.Bias = torch.Empty([]int{outChannels}, true)
		fanIn, _ := torch.CalculateFanInAndFanOut(c.Weight)
		bound := 1.0 / math.Sqrt(float64(fanIn))
		torch.Uniform(&c.Bias, -bound, bound)
	}
	return c
}

// Forward method
func (c *conv2d) Forward(x Tensor) Tensor {
	return functional.Conv2d(x, c.Weight, c.Bias, []int{c.Stride, c.Stride},
		[]int{c.Padding, c.Padding}, []int{c.Dilation, c.Dilation}, c.Groups)
}
