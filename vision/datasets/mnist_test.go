package datasets

import (
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func ExampleMNIST() {
	dataset := MNIST("", []transforms.Transform{transforms.Normalize(0.1307, 0.3081)})
	trainLoader := NewMNISTLoader(dataset, 8)
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
		MNIST(path.Join(os.TempDir(), "not_yet_exists"),
			[]transforms.Transform{})
	})
}
