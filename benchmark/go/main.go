package main

import (
	"fmt"
	"log"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
)

// MLPMNISTNet represents multiple layer perceptron neural network
type MLPMNISTNet struct {
	FC1, FC2, FC3 nn.Module
}

// Forward run the forward pass
func (n *MLPMNISTNet) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, []int64{-1, 28 * 28})
	x = n.FC1.Forward(x)
	x = torch.Tanh(x)
	x = n.FC2.Forward(x)
	x = torch.Tanh(x)
	x = n.FC3.Forward(x)
	return x.LogSoftmax(1)
}

// NewMNISTNet returns a MNIST module
func NewMNISTNet() nn.Module {
	return &MLPMNISTNet{
		FC1: nn.Linear(28*28, 512, false),
		FC2: nn.Linear(512, 512, false),
		FC3: nn.Linear(512, 10, false),
	}
}

func train() {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	net := NewMNISTNet()
	transforms := []torch.Transform{torch.NewNormalize(0.1307, 0.3081)}
	mnist := torch.NewMNIST(dataDir(), transforms)
	opt := torch.SGD(0.01, 0.5, 0, 0, false)
	opt.AddParameters(nn.GetParameters(net))
	epochs := 10
	// TODO(yancey1989): port dataset size API
	totalSamples := 60000
	totalThroughput := 0
	for epochIdx := 0; epochIdx < epochs; epochIdx++ {
		startTime := time.Now()
		trainLoader := torch.NewDataLoader(mnist, 64)
		batchIdx := 0
		for trainLoader.Scan() {
			batch := trainLoader.Batch()
			opt.ZeroGrad()
			pred := net.Forward(batch.Data)
			loss := F.NllLoss(pred, batch.Target, torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
			if batchIdx%200 == 0 {
				fmt.Printf("Train Epoch: %d, BatchIdx: %d, Loss: %s\n", epochIdx, batchIdx, loss)
			}
			batchIdx++
		}
		trainLoader.Close()
		endTime := time.Now()
		var duration float64 = endTime.Sub(startTime).Seconds()
		throughput := int(float64(totalSamples) / duration)
		totalThroughput += throughput
		fmt.Printf("End Train Epoch %d, Throughput: %d \n", epochIdx, throughput)
	}
	fmt.Printf("The average throughout: %d \n", totalThroughput/epochs)
	mnist.Close()
	torch.FinishGC()
}

func main() {
	train()
}
