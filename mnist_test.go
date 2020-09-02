package gotorch_test

import (
	"log"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/data"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

type MLPMNISTSequential struct {
	nn.Module
	Layers *nn.SequentialModule
}

func (s *MLPMNISTSequential) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, []int64{-1, 28 * 28})
	return s.Layers.Forward(x).(torch.Tensor).LogSoftmax(1)
}

func ExampleTrainMNISTSequential() {
	net := &MLPMNISTSequential{Layers: nn.Sequential(
		nn.Linear(28*28, 512, false),
		nn.Functional(torch.Tanh),
		nn.Linear(512, 512, false),
		nn.Functional(torch.Tanh),
		nn.Linear(512, 10, false))}
	net.Init(net)

	mnist := datasets.MNIST("",
		[]transforms.Transform{transforms.Normalize([]float64{0.1307}, []float64{0.3081})}, 64)

	opt := torch.SGD(0.1, 0.5, 0, 0, false)
	opt.AddParameters(net.Parameters())
	epochs := 1
	startTime := time.Now()
	for i := 0; i < epochs; i++ {
		for batch := range data.Loader(mnist) {
			opt.ZeroGrad()
			pred := net.Forward(batch.Data())
			loss := F.NllLoss(pred, batch.Target(), torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
		}
	}
	throughput := float64(60000*epochs) / time.Since(startTime).Seconds()
	log.Printf("Throughput: %f samples/sec", throughput)
	mnist.Close()
	torch.FinishGC()
	// Output:
}
