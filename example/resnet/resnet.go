package main

import (
	"fmt"
	"log"
	"math"
	"reflect"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

// BasicBlockModule struct
type BasicBlockModule struct {
	nn.Module
	C1, C2     *nn.Conv2dModule
	BN1, BN2   *nn.BatchNorm2dModule
	Downsample *nn.SequentialModule
}

// BasicBlock returns a BasicBlockModule instance
func BasicBlock(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) *BasicBlockModule {
	b := &BasicBlockModule{
		C1:         nn.Conv2d(inplanes, planes, 3, stride, 1, 1, 1, false, "zeros"),
		BN1:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		C2:         nn.Conv2d(planes, planes, 3, 1, 1, 1, 1, false, "zeros"),
		BN2:        nn.BatchNorm2d(planes, 1e-5, 0.1, true, true),
		Downsample: downsample,
	}
	b.Init(b)
	return b
}

// Forward method
func (b *BasicBlockModule) Forward(x torch.Tensor) torch.Tensor {
	identity := x

	out := b.C1.Forward(x)
	out = b.BN1.Forward(out)
	out = torch.Relu(out)

	out = b.C2.Forward(out)
	out = b.BN2.Forward(out)

	if b.Downsample != nil {
		identity = b.Downsample.Forward(x).(torch.Tensor)
	}

	out = torch.Add(out, identity, 1)
	out = torch.Relu(out)
	return out
}

// BottleneckModule struct
type BottleneckModule struct {
	nn.Module
	C1, C2, C3    *nn.Conv2dModule
	BN1, BN2, BN3 *nn.BatchNorm2dModule
	Downsample    *nn.SequentialModule
}

// Bottleneck returns a BottleneckModule instance
func Bottleneck(inplanes, planes, stride int64, downsample *nn.SequentialModule,
	groups, baseWidth, dilation int64) *BottleneckModule {
	width := (planes * baseWidth / 64) * groups
	expension := int64(4)
	b := &BottleneckModule{
		C1:         nn.Conv2d(inplanes, width, 1, 1, 0, 1, 1, false, "zeros"),
		C2:         nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, false, "zeros"),
		C3:         nn.Conv2d(width, planes*expension, 1, 1, 0, 1, 1, false, "zeros"),
		BN1:        nn.BatchNorm2d(width, 1e-5, 0.1, true, true),
		BN2:        nn.BatchNorm2d(width, 1e-5, 0.1, true, true),
		BN3:        nn.BatchNorm2d(planes*expension, 1e-5, 0.1, true, true),
		Downsample: downsample,
	}
	b.Init(b)
	return b
}

// Forward method
func (b *BottleneckModule) Forward(x torch.Tensor) torch.Tensor {
	identity := x
	out := b.C1.Forward(x)

	out = b.BN1.Forward(out)
	out = torch.Relu(out)

	out = b.C2.Forward(out)
	out = b.BN2.Forward(out)
	out = torch.Relu(out)

	out = b.C3.Forward(out)
	out = b.BN3.Forward(out)
	if b.Downsample != nil {
		identity = b.Downsample.Forward(x).(torch.Tensor)
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
	C1                                    *nn.Conv2dModule
	BN1                                   *nn.BatchNorm2dModule
	L1, L2, L3, L4                        *nn.SequentialModule
	FC                                    *nn.LinearModule
	Block                                 reflect.Type
	Inplanes, Groups, BaseWidth, Dilation int64
}

// Resnet returns a ResnetModule instance
func Resnet(block reflect.Type, layers []int64, numClasses int64, zeroInitResidual bool, groups, widthPerGroup int64) *ResnetModule {
	inplanes := int64(64)
	r := &ResnetModule{
		C1:        nn.Conv2d(3, inplanes, 7, 2, 3, 1, 1, false, "zeros"),
		BN1:       nn.BatchNorm2d(inplanes, 1e-5, 0.1, true, true),
		FC:        nn.Linear(512*getExpension(block), numClasses, true),
		Inplanes:  inplanes,
		Groups:    groups,
		BaseWidth: widthPerGroup,
		Dilation:  1,
	}
	r.L1 = r.makeLayer(block, 64, layers[0], 1, false)
	r.L2 = r.makeLayer(block, 128, layers[1], 2, false)
	r.L3 = r.makeLayer(block, 256, layers[2], 2, false)
	r.L4 = r.makeLayer(block, 512, layers[3], 2, false)
	r.Init(r)
	return r
}

func (r *ResnetModule) makeLayer(block reflect.Type, planes, blocks, stride int64, dilate bool) *nn.SequentialModule {
	var downsample *nn.SequentialModule
	previousDilation := r.Dilation
	if dilate {
		r.Dilation *= stride
		stride = 1
	}

	if stride != 1 || r.Inplanes != planes*getExpension(block) {
		downsample = nn.Sequential(nn.Conv2d(r.Inplanes, planes*getExpension(block), 1, stride, 0, 1, 1, false, "zeros"),
			nn.BatchNorm2d(planes*getExpension(block), 1e-5, 0.1, true, true))
	}

	layers := []nn.IModule{}
	layers = append(layers, createBlock(block, r.Inplanes, planes, stride, downsample,
		r.Groups, r.BaseWidth, previousDilation))
	r.Inplanes = planes * getExpension(block)
	for i := int64(1); i < blocks; i++ {
		layers = append(layers, createBlock(block, r.Inplanes, planes, 1, nil,
			r.Groups, r.BaseWidth, r.Dilation))
	}

	return nn.Sequential(layers...)
}

// Forward method
func (r *ResnetModule) Forward(x torch.Tensor) torch.Tensor {
	x = r.C1.Forward(x)
	x = r.BN1.Forward(x)
	x = torch.Relu(x)
	x = F.MaxPool2d(x, []int64{3, 3}, []int64{2, 2}, []int64{1, 1}, []int64{1, 1}, true)

	x = r.L1.Forward(x).(torch.Tensor)
	x = r.L2.Forward(x).(torch.Tensor)
	x = r.L3.Forward(x).(torch.Tensor)
	x = r.L4.Forward(x).(torch.Tensor)

	x = F.AdaptiveAvgPool2d(x, []int64{1, 1})
	x = torch.Flatten(x, 1, -1)
	x = r.FC.Forward(x)

	return x
}

// Resnet18 returns a Resnet18 network
func Resnet18() *ResnetModule {
	return Resnet(reflect.TypeOf((*BasicBlockModule)(nil)).Elem(), []int64{2, 2, 2, 2}, 1000, false, 1, 64)
}

// Resnet50 returns a Resnet50 network
func Resnet50() *ResnetModule {
	return Resnet(reflect.TypeOf((*BottleneckModule)(nil)).Elem(), []int64{3, 4, 6, 3}, 1000, false, 1, 64)
}

func adjustLearningRate(opt torch.Optimizer, epoch int, lr float64) {
	newLR := lr * math.Pow(0.1, float64(epoch)/30.0)
	opt.SetLR(newLR)
}

func main() {
	batchSize := int64(16)
	epochs := 90
	lr := 0.1
	momentum := 0.9
	weightDecay := 1e-4

	var device torch.Device
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	model := Resnet50()
	model.To(device)

	optimizer := torch.SGD(lr, momentum, 0, weightDecay, false)
	optimizer.AddParameters(model.Parameters())

	for epoch := 0; epoch < epochs; epoch++ {
		adjustLearningRate(optimizer, epoch, lr)

		model.Train(true)
		{
			torch.GC()
			image := torch.RandN([]int64{batchSize, 3, 224, 224}, false).To(device, torch.Float)
			target := torch.Empty([]int64{batchSize}, false)
			initializer.Uniform(&target, 0, 1000)
			target = target.To(device, torch.Long)

			output := model.Forward(image)
			loss := F.CrossEntropy(output, target, torch.Tensor{}, -100, "mean")

			fmt.Printf("loss: %f\n", loss.Item())

			optimizer.ZeroGrad()
			loss.Backward()
			optimizer.Step()
		}
	}
	torch.FinishGC()
}
