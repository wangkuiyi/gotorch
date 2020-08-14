package gotorch_test

import (
	"log"
	"testing"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision"
)

func ExampleMNIST() {
	if e := vision.DownloadMNIST(); e != nil {
		log.Printf("Cannot find or download MNIST dataset: %v", e)
	}

	dataset := torch.NewMNIST(vision.MNISTDir(), []torch.Transform{torch.NewNormalize(0.1307, 0.3081)})
	trainLoader := torch.NewDataLoader(dataset, 8)
	for trainLoader.Scan() {
		_ = trainLoader.Batch()
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
	torch.NewMNIST("nonexist", []torch.Transform{})
}
