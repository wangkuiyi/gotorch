package main

import (
	"encoding/gob"
	"flag"
	"log"
	"math"
	"os"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

// the original Imagenet dataset contains 1281167 training images
// c.f. https://patrykchrabaszcz.github.io/Imagenet32/
const (
	trainSamples = 1281167
	logInterval  = 10 // in iterations
)

var device torch.Device

func maxIntSlice(v []int64) int64 {
	if len(v) == 0 {
		panic("maxIntSlice expected a non-empty slice")
	}
	max := v[0]
	for _, e := range v {
		if e > max {
			max = e
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
	maxk := maxIntSlice(topk)
	target = target.Detach()
	output = output.Detach()

	mbSize := target.Shape()[0]
	_, pred := torch.TopK(output, maxk, 1, true, true)
	pred = pred.Transpose(0, 1)
	correct := pred.Eq(target.View(1, -1).ExpandAs(pred))

	res := []float32{}
	for _, k := range topk {
		kt := torch.NewTensor(rangeI(k)).CopyTo(device)
		correctK := correct.IndexSelect(0, kt).View(-1).CastTo(torch.Float).Sum(map[string]interface{}{"dim": 0, "keepDim": true})
		res = append(res, correctK.Item().(float32)*100/float32(mbSize))
	}
	return res
}

func imageNetLoader(fn string, vocab map[string]int, mbSize int) *datasets.ImageLoader {
	trans := transforms.Compose(
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(0.5),
		transforms.ToTensor(),
		transforms.Normalize([]float64{0.485, 0.456, 0.406}, []float64{0.229, 0.224, 0.225}))

	loader, e := datasets.NewImageLoader(fn, vocab, trans, mbSize)
	if e != nil {
		log.Fatal(e)
	}
	return loader
}

func trainOneMinibatch(image, target torch.Tensor, model *models.ResnetModule, opt torch.Optimizer) (float32, float32, float32) {
	output := model.Forward(image)
	loss := F.CrossEntropy(output, target, torch.Tensor{}, -100, "mean")
	acc := accuracy(output, target, []int64{1, 5})
	acc1 := acc[0]
	acc5 := acc[1]
	loss.Backward()
	opt.Step()
	return loss.Item().(float32), acc1, acc5
}

func test(model *models.ResnetModule, loader *datasets.ImageLoader) {
	testLoss := float32(0)
	acc1 := float32(0)
	acc5 := float32(0)
	correct := int64(0)
	samples := 0
	for loader.Scan() {
		data, label := loader.Minibatch()
		data = data.To(device, data.Dtype())
		label = label.To(device, label.Dtype())
		output := model.Forward(data)
		acc := accuracy(output, label, []int64{1, 5})
		acc1 += acc[0]
		acc5 += acc[1]
		loss := F.CrossEntropy(output, label, torch.Tensor{}, -100, "mean")
		pred := output.Argmax(1)
		testLoss += loss.Item().(float32)
		correct += pred.Eq(label.View(pred.Shape()...)).Sum(map[string]interface{}{"dim": 0, "keepDim": false}).Item().(int64)
		samples += int(label.Shape()[0])
	}
	log.Printf("Test average loss: %.4f acc1: %.4f acc5: %.4f \n",
		testLoss/float32(samples), acc1/float32(samples), acc5/float32(samples))
}

func train(trainFn, testFn, save string, epochs int) {
	// build label vocabulary
	vocab, e := datasets.BuildLabelVocabularyFromTgz(trainFn)
	if e != nil {
		log.Fatal(e)
	}
	log.Print("building label vocabulary done.")
	model := models.Resnet50()
	model.To(device)
	model.Train(true)

	lr := 0.1
	momentum := 0.9
	weightDecay := 1e-4
	mbSize := 32
	optimizer := torch.SGD(lr, momentum, 0, weightDecay, false)
	optimizer.AddParameters(model.Parameters())

	for epoch := 0; epoch < epochs; epoch++ {
		adjustLearningRate(optimizer, epoch, lr)
		startTime := time.Now()
		trainLoader := imageNetLoader(trainFn, vocab, mbSize)
		testLoader := imageNetLoader(testFn, vocab, mbSize)
		iter := 0
		for trainLoader.Scan() {
			iter++
			data, label := trainLoader.Minibatch()
			optimizer.ZeroGrad()
			loss, acc1, acc5 := trainOneMinibatch(data.To(device, data.Dtype()), label.To(device, label.Dtype()), model, optimizer)
			if iter%logInterval == 0 {
				throughput := float64(data.Shape()[0]*logInterval) / time.Since(startTime).Seconds()
				log.Printf("Train Epoch: %d, Iteration: %d, loss:%f, acc1: %f, acc5:%f, throughput: %f samples/sec", epoch, iter, loss, acc1, acc5, throughput)
				startTime = time.Now()
			}
		}
		torch.FinishGC()
		test(model, testLoader)
	}
	saveModel(model, save)
}

func saveModel(model *models.ResnetModule, modelFn string) {
	log.Println("Saving model to", modelFn)
	f, e := os.Create(modelFn)
	if e != nil {
		log.Fatalf("Cannot create file to save model: %v", e)
	}
	defer f.Close()

	d := torch.NewDevice("cpu")
	model.To(d)
	if e := gob.NewEncoder(f).Encode(model.StateDict()); e != nil {
		log.Fatal(e)
	}
}

func main() {
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(1)
	trainTar := flag.String("data", "/tmp/imagenet_training_shuffled.tar.gz", "data tarball")
	testTar := flag.String("test", "/tmp/imagenet_testing_shuffled.tar.gz", "data tarball")
	save := flag.String("save", "/tmp/imagenet_model.gob", "the model file")
	epochs := flag.Int("epochs", 5, "the number of epochs")

	flag.Parse()

	train(*trainTar, *testTar, *save, *epochs)
}
