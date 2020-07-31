package main

import (
	"fmt"

	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	dataset := torch.NewMnist("./data")
	dataset.AddTransforms([]torch.Transform{
		torch.NewNormalize(0.1307, 0.3081),
		torch.NewStack(),
	})
	trainLoader := torch.NewDataLoader(dataset, 8)
	for trainLoader.Scan() {
		data := trainLoader.Batch().Data
		target := trainLoader.Batch().Target
		fmt.Println(data)
		fmt.Println(target)
	}
}
