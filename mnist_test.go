package gotorch_test

import (
	"fmt"
	"log"
	"testing"

	torch "github.com/wangkuiyi/gotorch"
)

type MultiLayerMNISTNet struct {
	FC1, FC2, FC3 torch.Module
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

func NewMNISTNet() torch.Module {
	return &MultiLayerMNISTNet{
		FC1: torch.Linear(28*28, 512, false),
		FC2: torch.Linear(512, 512, false),
		FC3: torch.Linear(512, 10, false),
	}
}

func TestTrainMNIST(t *testing.T) {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	net := NewMNISTNet()
	transforms := []torch.Transform{torch.NewNormalize(0.1307, 0.3081)}
	mnist := torch.NewMNIST(dataDir(), transforms)
	trainLoader := torch.NewDataLoader(mnist, 64)
	opt := torch.SGD(0.1, 0, 0, 0, false)
	opt.AddParameters(torch.GetParameters(net))
	batchIdx := 0
	for trainLoader.Scan() {
		batch := trainLoader.Batch()
		pred := net.Forward(batch.Data)
		loss := torch.CrossEntropyLoss(pred, batch.Target)
		loss.Backward()
		opt.ZeroGrad()
		batchIdx++
		if batchIdx%100 == 0 {
			fmt.Printf("batch: %d, Loss: %s", batchIdx, loss)
		}
	}
	trainLoader.Close()
	mnist.Close()
	torch.FinishGC()
}
