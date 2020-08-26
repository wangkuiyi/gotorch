package main

import (
	"fmt"
	"log"

	torch "github.com/wangkuiyi/gotorch"
	nn "github.com/wangkuiyi/gotorch/nn"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/nn/initializer"
	"github.com/wangkuiyi/gotorch/vision/datasets"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

var device torch.Device

func generator(nz int64) *nn.SequentialModule {
	return nn.Sequential(
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
	)
}

func discriminator() *nn.SequentialModule {
	return nn.Sequential(
		nn.Conv2d(1, 64, 4, 2, 1, 1, 1, false, "zeros"),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),
		nn.Conv2d(64, 128, 4, 2, 1, 1, 1, false, "zeros"),
		nn.BatchNorm2d(128, 1e-5, 0.1, true, true),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),
		nn.Conv2d(128, 256, 4, 2, 1, 1, 1, false, "zeros"),
		nn.BatchNorm2d(256, 1e-5, 0.1, true, true),
		nn.Functional(func(in torch.Tensor) torch.Tensor { return torch.LeakyRelu(in, 0.2) }),
		nn.Conv2d(256, 1, 3, 1, 0, 1, 1, false, "zeros"),
		nn.Functional(torch.Sigmoid),
	)
}

func main() {
	if torch.IsCUDAAvailable() {
		log.Println("CUDA is valid")
		device = torch.NewDevice("cuda")
	} else {
		log.Println("No CUDA found; CPU only")
		device = torch.NewDevice("cpu")
	}

	mnist := datasets.MNIST("",
		[]transforms.Transform{transforms.Normalize([]float64{0.5}, []float64{0.5})})

	nz := int64(100)
	lr := 0.0002

	netG := generator(nz)
	netG.To(device)
	netD := discriminator()
	netD.To(device)

	optimizerD := torch.Adam(lr, 0.5, 0.5, 0.0)
	optimizerD.AddParameters(netD.Parameters())

	optimizerG := torch.Adam(lr, 0.5, 0.5, 0.0)
	optimizerG.AddParameters(netG.Parameters())

	epochs := 30
	checkpointStep := 1000
	checkpointCount := 1
	batchSize := int64(64)
	i := 0
	for epoch := 0; epoch < epochs; epoch++ {
		trainLoader := datasets.NewMNISTLoader(mnist, batchSize)
		for trainLoader.Scan() {
			// (1) update D network
			// train with real
			optimizerD.ZeroGrad()

			batch := trainLoader.Batch()
			data := batch.Data.CopyTo(device)
			label := torch.Empty([]int64{batch.Data.Shape()[0]}, false).CopyTo(device)
			initializer.Uniform(&label, 0.8, 1.0)
			output := netD.Forward(data).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
			errDReal := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errDReal.Backward()

			// train with fake
			noise := torch.RandN([]int64{batch.Data.Shape()[0], nz, 1, 1}, false).CopyTo(device)
			fake := netG.Forward(noise).(torch.Tensor)
			initializer.Zeros(&label)
			output = netD.Forward(fake.Detach()).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
			errDFake := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errDFake.Backward()
			errD := errDReal.Item() + errDFake.Item()
			optimizerD.Step()

			// (2) update G network
			optimizerG.ZeroGrad()
			initializer.Ones(&label)
			output = netD.Forward(fake).(torch.Tensor).View([]int64{-1, 1}).Squeeze(1)
			errG := F.BinaryCrossEntropy(output, label, torch.Tensor{}, "mean")
			errG.Backward()
			optimizerG.Step()

			fmt.Printf("[%d/%d][%d] D_Loss: %f G_Loss: %f\n",
				epoch, epochs, i, errD, errG.Item())
			if i%checkpointStep == 0 {
				samples := netG.Forward(torch.RandN([]int64{10, nz, 1, 1}, false).CopyTo(device)).(torch.Tensor)
				ckName := fmt.Sprintf("dcgan-sample-%d.pt", checkpointCount)
				samples.Detach().Save(ckName)
				checkpointCount++
			}
			i++
		}
		trainLoader.Close()
	}
	mnist.Close()
	torch.FinishGC()
}
