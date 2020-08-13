package gotorch_test

import (
	"log"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

type MLPMNISTNet struct {
	nn.Module
	FC1, FC2, FC3 *nn.LinearModule
}

func NewMNISTNet() *MLPMNISTNet {
	r := &MLPMNISTNet{
		FC1: nn.Linear(28*28, 512, false),
		FC2: nn.Linear(512, 512, false),
		FC3: nn.Linear(512, 10, false)}
	r.Init(r)
	return r
}

func (n *MLPMNISTNet) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, []int64{-1, 28 * 28})
	x = n.FC1.Forward(x)
	x = torch.Tanh(x)
	x = n.FC2.Forward(x)
	x = torch.Tanh(x)
	x = n.FC3.Forward(x)
	return x.LogSoftmax(1)
}

func ExampleTrainMNIST() {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	initializer.ManualSeed(1)
	transforms := []torch.Transform{torch.NewNormalize(0.1307, 0.3081)}
	mnist := torch.NewMNIST(dataDir(), transforms)

	net := NewMNISTNet()
	net.ZeroGrad()
	opt := torch.SGD(0.01, 0.5, 0, 0, false)
	opt.AddParameters(net.Parameters())

	epochs := 5
	startTime := time.Now()
	var lastLoss float32
	for epoch := 0; epoch < epochs; epoch++ {
		trainLoader := torch.NewDataLoader(mnist, 64)
		for trainLoader.Scan() {
			batch := trainLoader.Batch()
			opt.ZeroGrad()
			pred := net.Forward(batch.Data)
			loss := F.NllLoss(pred, batch.Target, torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
			lastLoss = loss.Item()
		}
		log.Printf("Epoch: %d, Loss: %.4f", epoch, lastLoss)
		trainLoader.Close()
	}
	throughput := float64(60000*epochs) / time.Since(startTime).Seconds()
	log.Printf("Throughput: %f samples/sec", throughput)

	mnist.Close()
	torch.FinishGC()
	// Output:
}

type MLPMNISTSequential struct {
	nn.Module
	Layers *nn.SequentialModule
}

func (s *MLPMNISTSequential) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, []int64{-1, 28 * 28})
	return s.Layers.Forward(x).(torch.Tensor).LogSoftmax(1)
}

func ExampleTrainMNISTSequential() {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	net := &MLPMNISTSequential{Layers: nn.Sequential(
		nn.Linear(28*28, 512, false),
		nn.Functional(torch.Tanh),
		nn.Linear(512, 512, false),
		nn.Functional(torch.Tanh),
		nn.Linear(512, 10, false))}
	net.Init(net)
	net.ZeroGrad()
	transforms := []torch.Transform{torch.NewNormalize(0.1307, 0.3081)}
	mnist := torch.NewMNIST(dataDir(), transforms)
	opt := torch.SGD(0.1, 0.5, 0, 0, false)
	opt.AddParameters(net.Parameters())
	epochs := 1
	startTime := time.Now()
	for i := 0; i < epochs; i++ {
		trainLoader := torch.NewDataLoader(mnist, 64)
		for trainLoader.Scan() {
			batch := trainLoader.Batch()
			opt.ZeroGrad()
			pred := net.Forward(batch.Data)
			loss := F.NllLoss(pred, batch.Target, torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
		}
		trainLoader.Close()
	}
	throughput := float64(60000*epochs) / time.Since(startTime).Seconds()
	log.Printf("Throughput: %f samples/sec", throughput)
	mnist.Close()
	torch.FinishGC()
	// Output:
}
