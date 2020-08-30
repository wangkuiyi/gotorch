package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"reflect"
	"strings"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/models"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func main() {
	defer torch.FinishGC()

	toTrain := flag.Bool("train", true, "Train or predict")
	modelFn := flag.String("model", "./mnist_model.gob", "Model filename")
	inputs := flag.String("inputs", "", "colon-separated input image files")
	flag.Parse()

	initializer.ManualSeed(1)

	var e error
	if *toTrain {
		e = train(*modelFn)
	} else {
		e = predict(*modelFn, *inputs)
	}
	if e != nil {
		log.Print(e)
	}
}

func train(modelFn string) error {
	device := defaultDevice()
	net := models.MLP()
	net.To(device)
	opt := torch.SGD(0.01, 0.5, 0, 0, false)
	opt.AddParameters(net.Parameters())

	mnist := datasets.MNIST("", []transforms.Transform{
		transforms.Normalize([]float64{0.1307}, []float64{0.3081})})
	defer mnist.Close()

	epochs := 5
	startTime := time.Now()
	var lastLoss float32
	iters := 0
	for epoch := 0; epoch < epochs; epoch++ {
		trainLoader := datasets.NewMNISTLoader(mnist, 64)
		for trainLoader.Scan() {
			batch := trainLoader.Batch()
			data := batch.Data.To(device, batch.Data.Dtype())
			target := batch.Target.To(device, batch.Target.Dtype())
			opt.ZeroGrad()
			pred := net.Forward(data)
			loss := F.NllLoss(pred, target, torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
			lastLoss = loss.Item()
			iters++
		}
		log.Printf("Epoch: %d, Loss: %.4f", epoch, lastLoss)
		trainLoader.Close()
	}
	throughput := float64(60000*epochs) / time.Since(startTime).Seconds()
	log.Printf("Throughput: %f samples/sec", throughput)

	return saveModel(net, modelFn)
}

func defaultDevice() torch.Device {
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		return torch.NewDevice("cuda")
	}
	log.Println("No CUDA found; CPU only")
	return torch.NewDevice("cpu")
}

func saveModel(model nn.IModule, modelFn string) error {
	f, e := os.Create(modelFn)
	if e != nil {
		return fmt.Errorf("Cannot create file to save model: %v", e)
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(model.StateDict())
}

func predict(modelFn, inputs string) error {
	m, e := loadModel(modelFn)
	if e != nil {
		return e
	}

	for _, fn := range strings.Split(inputs, ":") {
		f, e := os.Open(fn)
		if e != nil {
			return fmt.Errorf("Cannot open input file: %v", e)
		}
		defer f.Close()

		img, _, e := image.Decode(f)
		if e != nil {
			return fmt.Errorf("Cannot decode input image: %v", e)
		}

		fmt.Println(reflect.TypeOf(img))

		t := transforms.ToTensor().Run(img)
		n := transforms.Normalize([]float64{0.1307}, []float64{0.3081}).Run(t)
		fmt.Println(m.Forward(n))
	}
	return nil
}

func loadModel(modelFn string) (*models.MLPModule, error) {
	f, e := os.Open(modelFn)
	if e != nil {
		return nil, fmt.Errorf("Cannot open model file: %v", e)
	}
	defer f.Close()

	ns := make(map[string]torch.Tensor)
	if e := gob.NewDecoder(f).Decode(&ns); e != nil {
		return nil, fmt.Errorf("Cannot decode model: %v", e)
	}

	net := models.MLP()
	net.SetStateDict(ns)
	return net, nil
}
