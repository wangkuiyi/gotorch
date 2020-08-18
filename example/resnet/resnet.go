package main

import (
	"reflect"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
)

type BasicBlockModule struct {
	nn.Module
	conv1, conv2 *nn.Conv2dModule
	bn1, bn2     *nn.BatchNorm2dModule
	downsample   *nn.SequentialModule
}

func BasicBlock(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) *BasicBlockModule {
	b = &BasicBlockModule{
		conv1:      nn.Conv2d(inplanes, planes, 3, stride, 1, 1, 1, false, "zeros"),
		bn1:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		conv2:      nn.Conv2d(planes, planes, 3, stride, 1, 1, 1, false, "zeros"),
		bn2:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		downsample: downsample,
	}
	b.Init(b)
	return b
}

func (self *BasicBlockModule) Forward(x torch.Tensor) torch.Tensor {
	identity := x

	out := self.conv1.Forward(x)
	out = self.bn1.Forward(out)
	out = torch.Relu(out)

	out = self.conv2.Forward(out)
	out = self.bn2.Forward(out)

	if self.downsample != nil {
		identity = self.downsample.Forward(x).(torch.Tensor)
	}

	out = torch.Add(out, identity)
	out = torch.Relu(out)
	return out
}

type BottleneckModule struct {
	nn.Module
	conv1, conv2, conv3 *nn.Conv2dModule
	bn1, bn2, bn3       *nn.BatchNorm2dModule
	downsample          *nn.SequentialModule
}

func Bottleneck(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) *BottleneckModule {
	width := (planes * baseWidth / 64) * groups
	expension := 4
	b := &BottleneckModule{
		conv1:      nn.Conv2d(inplanes, width, 1, 1, 0, 1, 1, false, "zeros"),
		conv2:      nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, false, "zeros"),
		conv3:      nn.Conv2d(width, planes*expensions, 1, 1, 0, 1, 1, false, "zeros"),
		bn1:        nn.BatchNorm2d(width, 1e-5, 0.1, true, true),
		bn2:        nn.BatchNorm2d(width, 1e-5, 0.1, true, true),
		bn3:        nn.BatchNorm2d(planes*expensoions, 1e-5, 0.1, true, true),
		downsample: downsample,
	}
	b.Init(b)
	return b
}

func (self *BottleneckModule) Forward(x torch.Tensor) torch.Tensor {
	identity := x
	out := self.conv1.Forward(x)
	out = self.bn1.Forward(out)
	out = torch.Relu(out)

	out = self.conv2.Forward(x)
	out = self.bn2.Forward(x)
	out = torch.Relu(x)

	out = self.conv3.Forward(x)
	out = self.bn3.Forward(x)

	if self.downsample != nil {
		identity = self.downsample.Forward(x).(torch.Tensor)
	}

	out = torch.Add(out, identity)
	out = torch.Relu(out)
	return out
}


func getExpensions(t reflect.type) int64 {
	if t.Name() == "BasicBlockModule" {
		return 1
	}
	if t.Name() == "BottleneckModule" {
		return 4
	}
	panic("Unsupported Block Type")
}

func createBlock(t reflect.type, inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) nn.IModule {
	if t.Name() == "BasicBlockModule" {
		return BasicBlock(inplanes, planes, stride, downsample, groups, baseWidth, dilation)
	}
	if t.Name() == "BottleneckModule" {
		return Bottleneck(inplanes, planes, stride, downsample, groups, baseWidth, dilation)
	}
	panic("Unsupported Block Type")
}
type ResnetModule struct {
	nn.Module
	conv1                          *nn.Conv2dModule
	bn1                            *nn.BatchNorm2dModule
	layer1, layer2, layer3, layer4 *nn.SequentialModule
	fc                             *nn.LinearModule
	block                          reflect.Type
	inplanes, groups, baseWidth, dilation int64
}

func Resnet(block reflect.Type, layers []int64, numClasses int64, zeroInitResidual bool, groups int64, widthPerGroup int64) *ResnetModule {
	inplanes := 64
	r := &ResnetModule{
		conv1:  nn.Conv2d(3, inplanes, 7, 2, 3, 1, 1, false, "zeros"),
		bn1:    nn.BatchNorm2d(inplanes, 1e-5, 0.1, true, true),
		layer1: makeLayer(),
		layer2: makeLayer(),
		layer3: makeLayer(),
		layer4: makeLayer(),
	}
	r.inplanes = inplanes
	r.groups = groups
	r.baseWidth = widthPerGroup
	r.dilation = 1
	r.Init(r)
	return r
}

func (self *ResnetModule) makeLayer(block reflect.Type, planes, blocks, stride int64, dilate bool) *nn.SequentialModule {
	var downsample *nn.SequentialModule
	previousDilation := self.dilation
	if dilate {
		self.dilation *= stride
		stride = 1
	}

	if stride != 1 || self.inplanes != planes * getExpensions(block) {
		downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * getExpensions(block),1, stride, 0, 1, 1, false, "zeros"),
		nn.BatchNorm2d(planes * getExpensions(block), 1e-5, 0.1, true, true))
	}

	layers = []nn.IModule
	layers = append(layers, createBlock(block, self.inplanes, planes, stride, downsample,
		self.groups, self.baseWidth, previousDilation))

	for i := 1; i < blocks; i++ {
		layers = append(layers, createBlock(block, self.inplanes, planes, stride, downsample,
			self.groups, self.baseWidth, self.dilation))
	}

	return nn.Sequential(layers)
}

func (self *ResnetModule) Forward(x torch.Tensor) torch.Tensor {
	x = self.conv1.Forward(x)
	x = self.bn1.Forward(x)
	x = torch.Relu(x)
	x = F.MaxPool2d(x, []int64{3, 3}, []int64{2, 3}, []int64{1, 1}, []int64{1, 1}, true)

	x = self.layer1.Forward(x)
	x = self.layer2.Forward(x)
	x = self.layer3.Forward(x)
	x = self.layer4.Forward(x)

	x = F.AdaptiveAvgPool2d(x, []int64{1, 1})
	x = torch.Flatten(x, 0, -1)
	x = self.fc.Forward(x)

	return x
}

func Resnet50() {
	
}
