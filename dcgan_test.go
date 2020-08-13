package gotorch_test

import (
	"log"
	"time"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
)

type Generator struct {
	nn.Module
	main *nn.SequentialModule
}

func NewGenerator(nz, ngf, nc int64) *Generator {
	g := &Generator{
		main: nn.Sequential(
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
		),
	}
	g.Init(g)
	return g
}

func (g *Generator) Forward(x torch.Tensor) torch.Tensor {
	return g.main.Forward(x).(torch.Tensor)
}

type Discriminator struct {
	nn.Module
	main *nn.SequentialModule
}

func NewDiscriminator(ndf, nc int64) *Discriminator {
	d := &Discriminator{
		main: nn.Sequential(
			nn.Conv2d(nc, ndf, 4, 2, 1, 1, 1, false, "zeros"),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(ndf, ndf*2, 4, 2, 1, 1, 1, false, "zeros"),
			nn.BatchNorm2d(ndf*2, 1e-5, 0.1, true, true),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, 1, 1, false, "zeros"),
			nn.BatchNorm2d(ndf*4, 1e-5, 0.1, true, true),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, 1, 1, false, "zeros"),
			nn.BatchNorm2d(ndf*8, 1e-5, 0.1, true, true),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(ndf*8, 1, 4, 1, 0, 1, 1, false, "zeros"),
			nn.Functional(torch.Sigmoid),
		),
	}
	d.Init(d)
	return d
}

func (d *Discriminator) Forward(x torch.Tensor) torch.Tensor {
	return d.main.Forward(x).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
}

func ExampleTrainDCGAN() {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	transforms := []torch.Transform{torch.NewNormalize(0.1307, 0.3081)}
	mnist := torch.NewMNIST(dataDir(), transforms)

	nz := int64(100)
	ngf := int64(64)
	ndf := int64(64)
	nc := int64(1)
	lr := 0.0002
	beta1 := 0.5

	netG := NewGenerator(nz, ngf, nc)
	netD := NewDiscriminator(ndf, nc)

	optimizerD := torch.Adam(lr, beta1, 0.999, 0.0)
	optimizerD.AddParameters(netD.Parameters())

	optimizerG := torch.Adam(lr, beta1, 0.999, 0.0)
	optimizerG.AddParameters(netG.Parameters())

	epochs := 5
	startTime := time.Now()
	for i := 0; i < epochs; i++ {
		trainLoader := torch.NewDataLoader(mnist, 64)
		for trainLoader.Scan() {
			batch := trainLoader.Batch()

			// train with real
			optimizerD.ZeroGrad()

			opt.ZeroGrad()
			pred := net.Forward(batch.Data)
			loss := F.NllLoss(pred, batch.Target, torch.Tensor{}, -100, "mean")
			loss.Backward()
			opt.Step()
		}
		trainLoader.Close()
	}
	throughput := float64(60000*epochs) / time.Since(startTime).Seconds()
	log.Printf("Throughput: %f samples/sec", throughput)

	mnist.Close()
	torch.FinishGC()
	// Output:
}
