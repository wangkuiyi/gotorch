package gotorch_test

import (
	"testing"

	torch "github.com/wangkuiyi/gotorch"
)

func TestNotCompleteExampleMNIST(t *testing.T) {
	for i := 0; i < 20; i++ {
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
		torch.FinishGC()
	}
	// Output:
}
