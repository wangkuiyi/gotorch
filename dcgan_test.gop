package main

import (
	"crypto/rand"
	"flag"
	"log"
	"os"
	"strings"

	"github.com/gotorch/backends/cudnn"
	dset "github.com/gotorch/torchvision/datasets"
	transforms "github.com/gotorch/torchvision/transforms"
	"github.com/gotorch/utils/data"
)

var (
	dataset    = flag.String("dataset", "", "cifar10 | lsun | mnist |imagenet | folder | lfw | fake")
	dataroot   = flag.String("dataroot", "", "path to dataset")
	workers    = flag.Int("workers", 2, "number of data loading workers")
	batchSize  = flag.Int("batchSize", 64, "input batch size")
	imageSize  = flag.Int("imageSize", 64, "the height / width of the input image to network")
	nz         = flag.Int("nz", 100, "size of the latent z vector")
	ngf        = flag.Int("ngf", 64, "")
	ndf        = flag.Int("ndf", 64, "")
	niter      = flag.Int("niter", 25, "number of epochs to train for")
	lr         = flag.Float64("lr", 0.0002, "learning rate, default=0.0002")
	beta       = flag.Float64("beta1", 0.5, "beta1 for adam. default=0.5")
	cuda       = flag.Bool("cuda", false, "enables cuda")
	dryRun     = flag.Bool("dry-run", false, "check a single training cycle works")
	ngpu       = flag.Int("ngpu", 1, "number of GPUs to use")
	netG       = flag.String("netG", "", "path to netG (to continue training)")
	netD       = flag.String("netD", "", "path to netD (to continue training)")
	outf       = flag.String("outf", ".", "folder to output images and model checkpoints")
	manualSeed = flag.Int("manualSeed", 0, "manual seed")
	classes    = flag.String("classes", "bedroom", "comma separated list of classes for the lsun data set")
)

func main() {
	flag.Parse()

	os.MkdirAll(*outf)

	if *manualSeed != 0 {
		torch.ManualSeed(*manualSeed)
	} else {
		torch.ManualSeed(rand.Int())
	}

	cudnn.SetBenchmark(true)

	var (
		dataset dset.DataSet
		nc      int
	)

	switch *dataset {
	case "imagenet", "folder", "lfw":
		dataset = dset.ImageFolder(*dataroot,
			// Go+ doesn't need []transforms.Transformer in the following line.
			transforms.Compose([]transforms.Transformer{
				transforms.Resize(*imageSize),
				transforms.CenterCrop(*imageSize),
				transforms.ToTensor(),
				// Go+ doesn't need []float64 in the following line.
				transforms.Normalize([]float64{0.5, 0.5, 0.5}, []float64{0.5, 0.5, 0.5}),
			}))
		nc = 3
	case "lsun":
		strings.Split(*classes).Map(func(x string) string { return x + "_train" })
		dataset = dset.LSUN(*dataroot, classes,
			transforms.Compose([]transforms.Transformer{
				transforms.Resize(*imageSize),
				transforms.CenterCrop(*imageSize),
				transforms.ToTensor(),
				transforms.Normalize([]float64{0.5, 0.5, 0.5}, []float64{0.5, 0.5, 0.5}),
			}))
		nc = 3
	default:
		log.Fatalf("Unknown dataset %s", *dataset)
	}

	dataloader := data.NewLoader(dataset, *batchSize, true /*shuffle*/, *workers)
}
