package main

import (
	"flag"
	"io"
	"log"
	"math"
	"os"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

var trainSamples = 1281167
var device torch.Device

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
		kt := torch.NewTensor(rangeI(k)).CopyTo(device)
		correctK := correct.IndexSelect(0, kt).View([]int64{-1}).CastTo(torch.Float).SumByDim(0, true)
		res = append(res, correctK.Item()*100/float32(batchSize))
	}
	return res
}

func imageNetLoader(r io.Reader, vocab map[string]int64, batchSize int) (*datasets.ImageNetLoader, error) {
	trans := transforms.Compose(
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(0.5),
		transforms.ToTensor(),
		transforms.Normalize([]float64{0.485, 0.456, 0.406}, []float64{0.229, 0.224, 0.225}))

	loader, e := datasets.ImageNet(r, vocab, trans, batchSize)
	if e != nil {
		return nil, e
	}
	return loader, nil
}

func trainOneBatch(image, target torch.Tensor, model *models.ResnetModule, opt torch.Optimizer) (float32, float32, float32) {
	output := model.Forward(image)
	loss := F.CrossEntropy(output, target, torch.Tensor{}, -100, "mean")
	acc := accuracy(output, target, []int64{1, 5})
	acc1 := acc[0]
	acc5 := acc[1]
	loss.Backward()
	opt.Step()
	return loss.Item(), acc1, acc5
}

func main() {
	trainFile := flag.String("train-file", "train.tgz", "training images folder with tgz compressed.")
	flag.Parse()
	if _, e := os.Open(*trainFile); e != nil {
		panic(e)
	}

	batchSize := 32
	epochs := 100
	lr := 0.1
	momentum := 0.9
	weightDecay := 1e-4

	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	model := models.Resnet50()
	model.To(device)
	model.Train(true)

	optimizer := torch.SGD(lr, momentum, 0, weightDecay, false)
	optimizer.AddParameters(model.Parameters())

	f, e := os.Open(*trainFile)
	if e != nil {
		log.Fatal(e)
	}
	// build label vocabulary
	vocab, e := datasets.BuildLabelVocabulary(f)
	if e != nil {
		log.Fatal(e)
	}
	log.Print("building label vocabulary done.")

	itersEpoch := itersEachEpoch(trainSamples, batchSize)
	iter := 0
	epoch := 0
	adjustLearningRate(optimizer, epoch, lr)
	for {
		// reset file seeker and renew a ImageNet loader
		if _, e := f.Seek(0, io.SeekStart); e != nil {
			log.Fatal(e)
		}
		loader, e := imageNetLoader(f, vocab, batchSize)
		if e != nil {
			panic(e)
		}
		for loader.Scan() {
			iter++
			image, target := loader.Minibatch()
			loss, acc1, acc5 := trainOneBatch(image.CopyTo(device), target.CopyTo(device), model, optimizer)
			log.Printf("Epoch: %d, Batch: %d, loss:%f, acc1: %f, acc5:%f", epoch, iter, loss, acc1, acc5)
			if iter == itersEpoch {
				epoch++
				adjustLearningRate(optimizer, epoch, lr)
				iter = 0
				if epoch == epochs {
					return
				}
			}
		}
	}
}

func itersEachEpoch(samples, batchSize int) int {
	itersEachEpoch := trainSamples / batchSize
	if trainSamples%batchSize != 0 {
		itersEachEpoch++
	}
	return itersEachEpoch
}
