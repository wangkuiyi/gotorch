package gotorch_test

import (
	"testing"

	torch "github.com/wangkuiyi/gotorch"
)

func NotCompleteExampleMNIST(t *testing.T) {
	dataset := torch.NewMNIST("./data")
	dataset.AddTransforms([]torch.Transform{
		torch.NewNormalize(0.1307, 0.3081),
		torch.NewStack(),
	})
	trainLoader := torch.NewDataLoader(dataset, 8)
	for trainLoader.Scan() {
		torch.GC()
		trainLoader.Data()
	}
	trainLoader.Close()
	dataset.Close()
	torch.FinishGC()
	// Output:
}
