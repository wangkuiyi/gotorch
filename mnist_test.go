package gotorch_test

import (
	"fmt"
	"log"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
)

type MultiLayerMNISTNet struct {
	FC1, FC2, FC3 nn.Module
}

func (n *MultiLayerMNISTNet) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, []int{-1, 28 * 28})
	x = n.FC1.Forward(x)
	x = torch.Relu(x)
	x = n.FC2.Forward(x)
	x = torch.Relu(x)
	x = n.FC3.Forward(x)
	return x.LogSoftmax(1)
}

func NewMNISTNet() nn.Module {
	return &MultiLayerMNISTNet{
		FC1: nn.Linear(28*28, 512, false),
		FC2: nn.Linear(512, 512, false),
		FC3: nn.Linear(512, 10, false),
	}
}

func ExampleTrainMNIST() {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	net := NewMNISTNet()
	transforms := []torch.Transform{torch.NewNormalize(0.1307, 0.3081)}
	mnist := torch.NewMNIST(dataDir(), transforms)
	trainLoader := torch.NewDataLoader(mnist, 64)
	opt := torch.SGD(0.1, 0, 0, 0, false)
	opt.AddParameters(nn.GetParameters(net))
	batchIdx := 0
	for trainLoader.Scan() {
		batch := trainLoader.Batch()
		pred := net.Forward(batch.Data)
		loss := F.NllLoss(pred, batch.Target, torch.Tensor{nil}, -100, "mean")
		opt.ZeroGrad()
		loss.Backward()
		batchIdx++
		if batchIdx%100 == 0 {
			fmt.Printf("batch: %d, Loss: %s", batchIdx, loss)
		}
	}
	trainLoader.Close()
	mnist.Close()
	torch.FinishGC()
}
