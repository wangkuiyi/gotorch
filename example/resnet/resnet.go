package main

import (
	"fmt"
	"log"
	"math"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/models"
)

func adjustLearningRate(opt torch.Optimizer, epoch int, lr float64) {
	newLR := lr * math.Pow(0.1, float64(epoch)/30.0)
	opt.SetLR(newLR)
}

func main() {
	batchSize := int64(16)
	epochs := 100
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

	model := models.Resnet50()
	model.To(device)

	optimizer := torch.SGD(lr, momentum, 0, weightDecay, false)
	optimizer.AddParameters(model.Parameters())

	start := time.Now()
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
	fmt.Println(time.Since(start).Seconds())
}
