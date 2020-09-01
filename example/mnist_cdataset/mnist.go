package main

import (
	"log"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/data"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func main() {
	var device torch.Device
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(1)

	mnist := datasets.MNIST("",
		[]transforms.Transform{transforms.Normalize([]float64{0.1307}, []float64{0.3081})}, 64)

	net := models.MLP()
	net.To(device)
	opt := torch.SGD(0.01, 0.5, 0, 0, false)
	opt.AddParameters(net.Parameters())

	epochs := 5
	startTime := time.Now()
	var lastLoss float32
	iters := 0
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := range data.Loader(mnist) {
			data, target := batch.Data().To(device), batch.Target().To(device)
			opt.ZeroGrad()
			pred := net.Forward(data)
			loss := F.NllLoss(pred, target, torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
			lastLoss = loss.Item().(float32)
			iters++
		}
		log.Printf("Epoch: %d, Loss: %.4f", epoch, lastLoss)
	}
	throughput := float64(60000*epochs) / time.Since(startTime).Seconds()
	log.Printf("Throughput: %f samples/sec", throughput)

	mnist.Close()
	torch.FinishGC()
}
