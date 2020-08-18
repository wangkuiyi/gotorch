package main

import (
	"fmt"
	"reflect"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
)

// BasicBlockModule struct
type BasicBlockModule struct {
	nn.Module
	conv1, conv2 *nn.Conv2dModule
	bn1, bn2     *nn.BatchNorm2dModule
	downsample   *nn.SequentialModule
}

// BasicBlock returns a BasicBlockModule instance
func BasicBlock(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) *BasicBlockModule {
	b := &BasicBlockModule{
		conv1:      nn.Conv2d(inplanes, planes, 3, stride, 1, 1, 1, false, "zeros"),
		bn1:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		conv2:      nn.Conv2d(planes, planes, 3, stride, 1, 1, 1, false, "zeros"),
		bn2:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		downsample: downsample,
	}
	b.Init(b)
	return b
}

// Forward method
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

	out = torch.Add(out, identity, 1)
	out = torch.Relu(out)
	return out
}

// BottleneckModule struct
type BottleneckModule struct {
	nn.Module
	conv1, conv2, conv3 *nn.Conv2dModule
	bn1, bn2, bn3       *nn.BatchNorm2dModule
	downsample          *nn.SequentialModule
}

// Bottleneck returns a BottleneckModule instance
func Bottleneck(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) *BottleneckModule {
	width := (planes * baseWidth / 64) * groups
	expension := int64(4)
	b := &BottleneckModule{
		conv1:      nn.Conv2d(inplanes, width, 1, 1, 0, 1, 1, false, "zeros"),
		conv2:      nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, false, "zeros"),
		conv3:      nn.Conv2d(width, planes*expension, 1, 1, 0, 1, 1, false, "zeros"),
		bn1:        nn.BatchNorm2d(width, 1e-5, 0.1, true, true),
		bn2:        nn.BatchNorm2d(width, 1e-5, 0.1, true, true),
		bn3:        nn.BatchNorm2d(planes*expension, 1e-5, 0.1, true, true),
		downsample: downsample,
	}
	b.Init(b)
	return b
}

// Forward method
func (b *BottleneckModule) Forward(x torch.Tensor) torch.Tensor {
	identity := x
	out := b.conv1.Forward(x)
	out = b.bn1.Forward(out)
	out = torch.Relu(out)

	out = b.conv2.Forward(x)
	out = b.bn2.Forward(x)
	out = torch.Relu(x)

	out = b.conv3.Forward(x)
	out = b.bn3.Forward(x)

	if b.downsample != nil {
		identity = b.downsample.Forward(x).(torch.Tensor)
	}

	out = torch.Add(out, identity, 1)
	out = torch.Relu(out)
	return out
}

func getExpension(t reflect.Type) int64 {
	if t.Name() == "BasicBlockModule" {
		return 1
	}
	if t.Name() == "BottleneckModule" {
		return 4
	}
	panic("Unsupported Block Type")
}

func createBlock(t reflect.Type, inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) nn.IModule {
	if t.Name() == "BasicBlockModule" {
		return BasicBlock(inplanes, planes, stride, downsample, groups, baseWidth, dilation)
	}
	if t.Name() == "BottleneckModule" {
		return Bottleneck(inplanes, planes, stride, downsample, groups, baseWidth, dilation)
	}
	panic("Unsupported Block Type")
}

// ResnetModule struct
type ResnetModule struct {
	nn.Module
	conv1                                 *nn.Conv2dModule
	bn1                                   *nn.BatchNorm2dModule
	layer1, layer2, layer3, layer4        *nn.SequentialModule
	fc                                    *nn.LinearModule
	block                                 reflect.Type
	inplanes, groups, baseWidth, dilation int64
}

// Resnet returns a ResnetModule instance
func Resnet(block reflect.Type, layers []int64, numClasses int64, zeroInitResidual bool, groups int64, widthPerGroup int64) *ResnetModule {
	inplanes := int64(64)
	r := &ResnetModule{
		conv1:     nn.Conv2d(3, inplanes, 7, 2, 3, 1, 1, false, "zeros"),
		bn1:       nn.BatchNorm2d(inplanes, 1e-5, 0.1, true, true),
		fc:        nn.Linear(512*getExpension(block), numClasses, true),
		inplanes:  inplanes,
		groups:    groups,
		baseWidth: widthPerGroup,
		dilation:  1,
	}
	r.layer1 = r.makeLayer(block, 64, layers[0], 1, false)
	r.layer2 = r.makeLayer(block, 128, layers[1], 2, false)
	r.layer3 = r.makeLayer(block, 256, layers[2], 2, false)
	r.layer4 = r.makeLayer(block, 512, layers[3], 2, false)
	r.Init(r)
	return r
}

func (r *ResnetModule) makeLayer(block reflect.Type, planes, blocks, stride int64, dilate bool) *nn.SequentialModule {
	var downsample *nn.SequentialModule
	previousDilation := r.dilation
	if dilate {
		r.dilation *= stride
		stride = 1
	}

	if stride != 1 || r.inplanes != planes*getExpension(block) {
		downsample = nn.Sequential(nn.Conv2d(r.inplanes, planes*getExpension(block), 1, stride, 0, 1, 1, false, "zeros"),
			nn.BatchNorm2d(planes*getExpension(block), 1e-5, 0.1, true, true))
	}

	layers := []nn.IModule{}
	layers = append(layers, createBlock(block, r.inplanes, planes, stride, downsample,
		r.groups, r.baseWidth, previousDilation))

	for i := int64(1); i < blocks; i++ {
		layers = append(layers, createBlock(block, r.inplanes, planes, stride, downsample,
			r.groups, r.baseWidth, r.dilation))
	}

	return nn.Sequential(layers...)
}

// Forward method
func (r *ResnetModule) Forward(x torch.Tensor) torch.Tensor {
	x = r.conv1.Forward(x)
	x = r.bn1.Forward(x)
	x = torch.Relu(x)
	x = F.MaxPool2d(x, []int64{3, 3}, []int64{2, 3}, []int64{1, 1}, []int64{1, 1}, true)

	x = r.layer1.Forward(x).(torch.Tensor)
	x = r.layer2.Forward(x).(torch.Tensor)
	x = r.layer3.Forward(x).(torch.Tensor)
	x = r.layer4.Forward(x).(torch.Tensor)

	x = F.AdaptiveAvgPool2d(x, []int64{1, 1})
	x = torch.Flatten(x, 0, -1)
	x = r.fc.Forward(x)

	return x
}

// Resnet50 returns a Resnet50 network
func Resnet50() *ResnetModule {
	return Resnet(reflect.TypeOf((*BottleneckModule)(nil)).Elem(), []int64{3, 4, 6, 3}, 1000, false, 1, 64)
}

func main() {
	resnet50 := Resnet50()
	fmt.Println(resnet50.inplanes)
}
