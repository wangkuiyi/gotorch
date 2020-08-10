package gotorch_test

import (
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/user"
	"path"
	"testing"

	torch "github.com/wangkuiyi/gotorch"
)

type MultiLayerMNISTNet struct {
	FC1, FC2, FC3 torch.Module
}

func NewMNIST() torch.Module {
	return &MultiLayerMNISTNet{
		FC1: torch.Linear(28*28, 512, false),
		FC2: torch.Linear(512, 512, false),
		FC3: torch.Linear(512, 10, false),
	}
}

func (n *MultiLayerMNISTNet) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, []int{-1, 28 * 28})
	x = n.FC1.Forward(x)
	x = torch.Relu(x)
	x = n.FC2.Forward(x)
	x = torch.Relu(x)
	x = n.FC3.Forward(x)
	return x.Softmax()
}

func ExampleMNIST() {
	if e := downloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}
	mnist := NewMNIST()

	dataset := torch.NewMNIST(dataDir())
	dataset.AddTransforms([]torch.Transform{
		torch.NewNormalize(0.1307, 0.3081),
		torch.NewStack(),
	})
	trainLoader := torch.NewDataLoader(dataset, 2)
	opt := torch.SGD(0.1, 0, 0, 0, false)
	opt.AddParameters(torch.GetParameters(mnist))
	batchIdx := 0
	for trainLoader.Scan() {
		batch := trainLoader.Batch()
		fmt.Println(batch.Data)
		pred := mnist.Forward(batch.Data)
		fmt.Println(pred)
		fmt.Println(batch.Target)
		break
		loss := torch.CrossEntropyLoss(pred, batch.Target)
		fmt.Println(loss)
		break
		loss.Backward()
		opt.ZeroGrad()
		fmt.Println(loss)
		batchIdx++
		if batchIdx == 10 {
			break
		}
	}
	trainLoader.Close()
	dataset.Close()
	torch.FinishGC()
	// Output:
}

func TestPanicMNIST(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("TestPanicMNIST should have paniced")
		}
	}()
	torch.NewMNIST("nonexist")
}

const (
	images = "train-images-idx3-ubyte"
	labels = "train-labels-idx1-ubyte"
	site   = "http://yann.lecun.com/exdb/mnist/"
)

func downloadMNIST() error {
	if e := downloadIfNotYet(images); e != nil {
		return e
	}
	if e := downloadIfNotYet(labels); e != nil {
		return e
	}
	return nil
}

func downloadIfNotYet(fn string) error {
	f := path.Join(dataDir(), fn)
	if !fileExists(f) {
		if e := download(site+fn+".gz", f); e != nil {
			return e
		}
	}
	return nil
}

func dataDir() string {
	u, e := user.Current()
	if e != nil {
		return "testdata/mnist"
	}
	return path.Join(u.HomeDir, ".cache/mnist")
}

func fileExists(fn string) bool {
	info, e := os.Stat(fn)
	if os.IsNotExist(e) {
		return false
	}
	return !info.IsDir()
}

func download(url, fn string) error {
	resp, e := http.Get(url)
	if e != nil {
		return e
	}
	defer resp.Body.Close()

	r, e := gzip.NewReader(resp.Body)
	if e != nil {
		return e
	}
	defer r.Close()

	if e := os.MkdirAll(dataDir(), 0744); e != nil {
		return e
	}

	f, e := os.Create(fn)
	if e != nil {
		return e
	}
	defer f.Close()

	_, e = io.Copy(f, r)
	if e != nil && e != io.EOF {
		return e
	}

	return nil
}
