package gotorch_test

import (
	"fmt"
	"log"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
)

type Generator struct {
	nn.Module
	Main *nn.SequentialModule
}

func NewGenerator(nz int64) *Generator {
	g := &Generator{
		Main: nn.Sequential(
			nn.ConvTranspose2d(nz, 256, 4, 1, 0, 0, 1, false, 1, "zero"),
			nn.BatchNorm2d(256, 1e-5, 0.1, true, true),
			nn.Functional(torch.Relu),
			nn.ConvTranspose2d(256, 128, 3, 2, 1, 0, 1, false, 1, "zero"),
			nn.BatchNorm2d(128, 1e-5, 0.1, true, true),
			nn.Functional(torch.Relu),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, 0, 1, false, 1, "zero"),
			nn.BatchNorm2d(64, 1e-5, 0.1, true, true),
			nn.Functional(torch.Relu),
			nn.ConvTranspose2d(64, 1, 4, 2, 1, 0, 1, false, 1, "zero"),
			nn.Functional(torch.Tanh),
		),
	}
	g.Init(g)
	return g
}

func (g *Generator) Forward(x torch.Tensor) torch.Tensor {
	return g.Main.Forward(x).(torch.Tensor)
}

type Discriminator struct {
	nn.Module
	Main *nn.SequentialModule
}

func NewDiscriminator() *Discriminator {
	d := &Discriminator{
		Main: nn.Sequential(
			nn.Conv2d(1, 64, 4, 2, 1, 1, 1, false, "zeros"),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(64, 128, 4, 2, 1, 1, 1, false, "zeros"),
			nn.BatchNorm2d(128, 1e-5, 0.1, true, true),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(128, 256, 4, 2, 1, 1, 1, false, "zeros"),
			nn.BatchNorm2d(256, 1e-5, 0.1, true, true),
			nn.LeakyRelu(0.2, false),
			nn.Conv2d(256, 1, 3, 1, 0, 1, 1, false, "zeros"),
			nn.Functional(torch.Sigmoid),
		),
	}
	d.Init(d)
	return d
}

func (d *Discriminator) Forward(x torch.Tensor) torch.Tensor {
	return d.Main.Forward(x).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
}

func ExampleTrainDCGAN() {
	fmt.Println("nihao")
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	transforms := []torch.Transform{torch.NewNormalize(0.5, 0.5)}
	mnist := torch.NewMNIST(dataDir(), transforms)

	nz := int64(100)
	lr := 0.0002

	netG := NewGenerator(nz)
	netD := NewDiscriminator()

	optimizerD := torch.Adam(lr, 0.5, 0.5, 0.0)
	optimizerD.AddParameters(netD.Parameters())

	optimizerG := torch.Adam(lr, 0.5, 0.5, 0.0)
	optimizerG.AddParameters(netG.Parameters())

	epochs := 5
	batchSize := int64(64)
	for epoch := 0; epoch < epochs; epoch++ {
		i := 0
		trainLoader := torch.NewDataLoader(mnist, batchSize)
		for trainLoader.Scan() {
			// (1) update D network
			// train with real
			batch := trainLoader.Batch()
			optimizerD.ZeroGrad()
			label := torch.Empty([]int64{batchSize}, false)
			initializer.Uniform(&label, 0.8, 1.0)

			output := netD.Forward(batch.Data)
			errDReal := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errDReal.Backward()
			DX := output.Mean().Item()

			// train with fake
			noise := torch.RandN([]int64{batchSize, nz, 1, 1}, false)
			fake := netG.Forward(noise)
			initializer.Zeros(&label)
			output = netD.Forward(fake.Detach())
			errDFake := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errDFake.Backward()
			DGZ1 := output.Mean().Item()
			errD := errDReal.Item() + errDFake.Item()
			optimizerD.Step()

			// (2) update G network
			optimizerG.ZeroGrad()
			initializer.Ones(&label)
			output = netD.Forward(fake)
			errG := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errG.Backward()
			DGZ2 := output.Mean().Item()
			optimizerG.Step()
			fmt.Printf("[%d/%d][%d] Loss_D: %f Loss_G: %f D(x): %f D(G(z)): %f / %f\n",
				epoch, epochs, i, errD, errG.Item(), DX, DGZ1, DGZ2)
			i++
		}
		trainLoader.Close()
	}
	mnist.Close()
	torch.FinishGC()
	// Output:
}
