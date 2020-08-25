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

func max(array []int64) int64 {
	max := array[0]
	for _, v := range array {
		if v > max {
			max = v
		}
	}
	return max
}

func rangeI(n int64) []int64 {
	res := []int64{}
	if n <= 0 {
		return res
	}
	for i := int64(0); i < n; i++ {
		res = append(res, i)
	}
	return res
}

func adjustLearningRate(opt torch.Optimizer, epoch int, lr float64) {
	newLR := lr * math.Pow(0.1, float64(epoch)/30.0)
	opt.SetLR(newLR)
}

func accuracy(output, target torch.Tensor, topk []int64) []float32 {
	maxk := max(topk)
	target = target.Detach()
	output = output.Detach()

	batchSize := target.Shape()[0]
	_, pred := torch.TopK(output, maxk, 1, true, true)
	pred = pred.Transpose(0, 1)
	correct := pred.Eq(target.View([]int64{1, -1}).ExpandAs(pred))

	res := []float32{}
	for _, k := range topk {
		kt := torch.NewTensor(rangeI(k))
		correctK := correct.IndexSelect(0, kt).View([]int64{-1}).CastTo(torch.Float).SumByDim(0, true)
		res = append(res, correctK.Item()*100/float32(batchSize))
	}
	return res
}

func train(model *models.ResnetModule, opt torch.Optimizer, batchSize int64, device torch.Device) {
	model.Train(true)
	for i := 0; i < 10; i++ {
		torch.GC()
		image := torch.RandN([]int64{batchSize, 3, 224, 224}, false).To(device, torch.Float)
		target := torch.Empty([]int64{batchSize}, false)
		initializer.Uniform(&target, 0, 1000)
		target = target.To(device, torch.Long)

		output := model.Forward(image)
		loss := F.CrossEntropy(output, target, torch.Tensor{}, -100, "mean")

		acc := accuracy(output, target, []int64{1, 5})
		acc1 := acc[0]
		acc5 := acc[1]
		if i%5 == 0 {
			fmt.Printf("loss: %f, acc1 :%f, acc5: %f\n", loss.Item(), acc1, acc5)
		}

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	torch.FinishGC()
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
		train(model, optimizer, batchSize, device)
	}
	fmt.Println(time.Since(start).Seconds())
}
