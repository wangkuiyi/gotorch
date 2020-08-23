package vision_test

import (
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision"
)

func ExampleMNIST() {
	dataset := vision.MNIST("", []vision.Transform{vision.Normalize(0.1307, 0.3081)})
	trainLoader := vision.NewMNISTLoader(dataset, 8)
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
		vision.MNIST(path.Join(os.TempDir(), "not_yet_exists"),
			[]vision.Transform{})
	})
}
