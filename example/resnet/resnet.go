package main

import (
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
)

type BasicBlockModule struct {
	nn.Module
	conv1, conv2 *nn.Conv2dModule
	bn1, bn2     *nn.BatchNorm2dModule
	downsample   *nn.SequentialModule
}

func BasicBlock(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth int64) *BasicBlockModule {
	return &BasicBlockModule{
		conv1:      nn.Conv2d(inplanes, planes, int64(3), stride, int64(1), int64(1), int64(1), false, "zeros"),
		bn1:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		conv2:      nn.Conv2d(planes, planes, int64(3), stride, int64(1), int64(1), int64(1), false, "zeros"),
		bn2:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		downsample: downsample,
	}
}

func (b *BasicBlockModule) Forward(x torch.Tensor) torch.Tensor {
	identity := x

	out := b.conv1.Forward(x)
	out = b.bn1.Forward(out)
	out = torch.Relu(out)

	out = b.conv2.Forward(out)
	out = b.bn2.Forward(out)

	if b.downsample != nil {
		identity = b.downsample.Forward(x).(torch.Tensor)
	}

	out += identity
	out = torch.Relu(out)
	return out
}
