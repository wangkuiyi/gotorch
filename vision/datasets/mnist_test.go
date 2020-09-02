package datasets

import (
	//	"os"
	//	"path"
	//	"testing"

	//	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/data"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func ExampleMNIST() {
	dataset := MNIST("", []transforms.Transform{transforms.Normalize([]float64{0.1307}, []float64{0.3081})}, 8)
	for batch := range data.Loader(dataset) {
		_, _ = batch.Data(), batch.Target()
	}
	dataset.Close()
	gotorch.FinishGC()
	// Output:
}

// disable temporarily
// func TestNoPanicMNIST(t *testing.T) {
// 	assert.NotPanics(t, func() {
// 		MNIST(path.Join(os.TempDir(), "not_yet_exists"),
// 			[]transforms.Transform{}, 8)
// 	})
// }
