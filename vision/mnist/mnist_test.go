package mnist_test

import (
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision"
	"github.com/wangkuiyi/gotorch/vision/mnist"
)

func ExampleMNIST() {
	dataset := mnist.NewDataset("", []vision.Transform{vision.Normalize(0.1307, 0.3081)})
	trainLoader := mnist.NewLoader(dataset, 8)
	for trainLoader.Scan() {
		_ = trainLoader.Batch()
	}
	trainLoader.Close()
	dataset.Close()
	gotorch.FinishGC()
	// Output:
}

func TestNoPanicMNIST(t *testing.T) {
	assert.NotPanics(t, func() {
		mnist.NewDataset(path.Join(os.TempDir(), "not_yet_exists"),
			[]vision.Transform{})
	})
}
