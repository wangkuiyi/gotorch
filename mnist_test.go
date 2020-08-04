package gotorch_test

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

func NotComplete_ExampleMNIST() {
	dataset := torch.NewMNIST("./data")
	dataset.AddTransforms([]torch.Transform{
		torch.NewNormalize(0.1307, 0.3081),
		torch.NewStack(),
	})
	trainLoader := torch.NewDataLoader(dataset, 8)
	for trainLoader.Scan() {
		data := trainLoader.Data()
		fmt.Println(data.Data)
		fmt.Println(data.Target)
	}
	// Output:
}
