package main

import (
	"flag"
	"log"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

var device torch.Device

// MNISTLoader returns a ImageLoader with MNIST training or testing tgz file
func MNISTLoader(fn string, vocab map[string]int64) *datasets.ImageLoader {
	trans := transforms.Compose(transforms.ToTensor(), transforms.Normalize([]float64{0.1307}, []float64{0.3081}))
	loader, e := datasets.NewImageLoader(fn, vocab, trans, 64)
	if e != nil {
		panic(e)
	}
	return loader
}

func test(model nn.IModule, loader *datasets.ImageLoader) {
	testLoss := float32(0)
	correct := int64(0)
	samples := 0
	for loader.Scan() {
		data, label := loader.Minibatch()
		data = data.To(device, data.Dtype())
		label = label.To(device, label.Dtype())
		output := model.(*models.MLPModule).Forward(data)
		loss := F.NllLoss(output, label, torch.Tensor{}, -100, "mean")
		pred := output.Argmax(1)
		testLoss += loss.Item().(float32)
		correct += pred.Eq(label.View(pred.Shape())).SumByDim(0, false).Item().(int64)
		samples += int(label.Shape()[0])
	}
	log.Printf("Test average loss: %.4f, Accuracy: %.2f%%\n",
		testLoss/float32(samples), 100.0*float32(correct)/float32(samples))
}

func train(trainFn string, testFn string, epochs int) {
	vocab, e := datasets.BuildLabelVocabularyFromTgz(trainFn)
	if e != nil {
		panic(e)
	}
	net := models.MLP()
	net.To(device)
	opt := torch.SGD(0.01, 0.5, 0, 0, false)
	opt.AddParameters(net.Parameters())
	defer torch.FinishGC()

	for epoch := 0; epoch < epochs; epoch++ {
		var trainLoss float32
		startTime := time.Now()
		trainLoader := MNISTLoader(trainFn, vocab)
		testLoader := MNISTLoader(testFn, vocab)
		totalSamples := 0
		for trainLoader.Scan() {
			data, label := trainLoader.Minibatch()
			totalSamples += int(data.Shape()[0])
			opt.ZeroGrad()
			pred := net.Forward(data.To(device, data.Dtype()))
			loss := F.NllLoss(pred, label.To(device, label.Dtype()), torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
			trainLoss = loss.Item().(float32)
		}
		throughput := float64(totalSamples) / time.Since(startTime).Seconds()
		log.Printf("Train Epoch: %d, Loss: %.4f, throughput: %f samples/sec", epoch, trainLoss, throughput)
		test(net, testLoader)
	}
}

func main() {
	trainFn := flag.String("train-file", "train.tgz", "training images tgz file.")
	testFn := flag.String("test-file", "test.tgz", "testing images tgz file")
	epochs := flag.Int("epochs", 5, "number of epochs to train")
	flag.Parse()

	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(1)
	train(*trainFn, *testFn, *epochs)
}
