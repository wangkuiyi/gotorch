package main

import (
	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	dataset := torch.NewMnist("./data")
	dataset.AddTransforms([]torch.Transform{
		torch.NewNormalize(0.1307, 0.3081),
		torch.NewStack(),
	})

	//TODO(yancey1989): implement torch.DataLoader
	//trainDataLoader := torch.NewDataLoader(dataset, 8)
	//for batch_idx, batch := range trainDataLoader() {
	//  ..
	//}
}
