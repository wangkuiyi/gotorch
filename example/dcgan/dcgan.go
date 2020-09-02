package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"reflect"
	"strings"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

var dataroot = flag.String("dataroot", "", "path to dataset")
var device torch.Device

func weightInit(m nn.IModule) {
	if strings.Contains(m.Name(), "Conv") {
		fv := reflect.ValueOf(m.(*nn.Module).Outer()).Elem()
		for i := 0; i < fv.NumField(); i++ {
			v := fv.Field(i)
			f := fv.Type().Field(i)
			if f.Name == "Weight" {
				w := v.Interface().(torch.Tensor)
				initializer.Normal(&w, 0.0, 0.02)
			}
		}
	} else if strings.Contains(m.Name(), "BatchNorm") {
		fv := reflect.ValueOf(m.(*nn.Module).Outer()).Elem()
		for i := 0; i < fv.NumField(); i++ {
			v := fv.Field(i)
			f := fv.Type().Field(i)
			if f.Name == "Weight" {
				w := v.Interface().(torch.Tensor)
				initializer.Normal(&w, 1.0, 0.02)
			} else if f.Name == "Bias" {
				w := v.Interface().(torch.Tensor)
				initializer.Zeros(&w)
			}
		}
	}
}

func generator(nz int64, nc int64, ngf int64) *nn.SequentialModule {
	return nn.Sequential(
		nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, 0, 1, false, 1, "zero"),
		nn.BatchNorm2d(ngf*8, 1e-5, 0.1, true, true),
		nn.Functional(torch.Relu),

		nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, 0, 1, false, 1, "zero"),
		nn.BatchNorm2d(ngf*4, 1e-5, 0.1, true, true),
		nn.Functional(torch.Relu),

		nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, 0, 1, false, 1, "zero"),
		nn.BatchNorm2d(ngf*2, 1e-5, 0.1, true, true),
		nn.Functional(torch.Relu),

		nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, 0, 1, false, 1, "zero"),
		nn.BatchNorm2d(ngf, 1e-5, 0.1, true, true),
		nn.Functional(torch.Relu),

		nn.ConvTranspose2d(ngf, nc, 4, 2, 1, 0, 1, false, 1, "zero"),
		nn.Functional(torch.Tanh),
	)
}

func discriminator(nc int64, ndf int64) *nn.SequentialModule {
	return nn.Sequential(
		nn.Conv2d(nc, ndf, 4, 2, 1, 1, 1, false, "zeros"),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),

		nn.Conv2d(ndf, ndf*2, 4, 2, 1, 1, 1, false, "zeros"),
		nn.BatchNorm2d(ndf*2, 1e-5, 0.1, true, true),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),

		nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, 1, 1, false, "zeros"),
		nn.BatchNorm2d(ndf*4, 1e-5, 0.1, true, true),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),

		nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, 1, 1, false, "zeros"),
		nn.BatchNorm2d(ndf*8, 1e-5, 0.1, true, true),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),

		nn.Conv2d(ndf*8, 1, 4, 1, 0, 1, 1, false, "zeros"),
		nn.Functional(torch.Sigmoid),
	)
}

func celebaLoader(dataroot string, vocab map[string]int64, mbSize int) *datasets.ImageLoader {
	trans := transforms.Compose(transforms.Resize(64),
		transforms.CenterCrop(64),
		transforms.ToTensor(),
		transforms.Normalize([]float64{0.5, 0.5, 0.5}, []float64{0.5, 0.5, 0.5}))
	loader, e := datasets.NewImageLoader(dataroot, vocab, trans, mbSize)
	if e != nil {
		panic(e)
	}
	return loader
}

func main() {
	logFile, _ := os.OpenFile("gotorch-dcgan.log", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0666)
	mw := io.MultiWriter(os.Stdout, logFile)
	log.SetOutput(mw)

	flag.Parse()
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	initializer.ManualSeed(999)

	nc := int64(3)
	nz := int64(100)
	ngf := int64(64)
	ndf := int64(64)
	lr := 0.0002
	epochs := 15
	checkpointStep := 500
	batchSize := 128

	fixedNoise := torch.RandN([]int64{64, nz, 1, 1}, false).CopyTo(device)

	netG := generator(nz, nc, ngf)
	netG.To(device)
	netG.Apply(weightInit)
	netD := discriminator(nc, ndf)
	netD.To(device)
	netD.Apply(weightInit)

	optimizerD := torch.Adam(lr, 0.5, 0.999, 0.0)
	optimizerD.AddParameters(netD.Parameters())

	optimizerG := torch.Adam(lr, 0.5, 0.999, 0.0)
	optimizerG.AddParameters(netG.Parameters())

	vocab, e := datasets.BuildLabelVocabularyFromTgz(*dataroot)
	if e != nil {
		panic(e)
	}

	i := 0
	for epoch := 0; epoch < epochs; epoch++ {
		trainLoader := celebaLoader(*dataroot, vocab, batchSize)
		for trainLoader.Scan() {
			// (1) update D network
			// train with real
			optimizerD.ZeroGrad()
			data, _ := trainLoader.Minibatch()
			data = data.CopyTo(device)
			label := torch.Empty([]int64{data.Shape()[0]}, false).CopyTo(device)
			initializer.Ones(&label)
			output := netD.Forward(data).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
			errDReal := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errDReal.Backward()

			// train with fake
			noise := torch.RandN([]int64{data.Shape()[0], nz, 1, 1}, false).CopyTo(device)
			fake := netG.Forward(noise).(torch.Tensor)
			initializer.Zeros(&label)
			output = netD.Forward(fake.Detach()).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
			errDFake := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errDFake.Backward()
			errD := errDReal.Item().(float32) + errDFake.Item().(float32)
			optimizerD.Step()

			// (2) update G network
			optimizerG.ZeroGrad()
			initializer.Ones(&label)
			output = netD.Forward(fake).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
			errG := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errG.Backward()
			optimizerG.Step()

			log.Printf("| %d/%d | Step: %d | Loss_D: %.4f | Loss_G: %.4f |\n",
				epoch, epochs, i, errD, errG.Item())
			if i%checkpointStep == 0 {
				samples := netG.Forward(fixedNoise).(torch.Tensor)
				ckName := fmt.Sprintf("gotorch-dcgan-sample-%d.pt", i)
				samples.Detach().Save(ckName)
			}
			i++
		}
	}
	torch.FinishGC()
}
