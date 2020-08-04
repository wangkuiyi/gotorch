package main

import (
	torch "github.com/wangkuiyi/gotorch"
)

func trainLoop() {
	dataset := torch.NewMNIST("./data")
	dataset.AddTransforms([]torch.Transform{
		torch.NewNormalize(0.1307, 0.3081),
		torch.NewStack(),
	})
	trainLoader := torch.NewDataLoader(dataset, 8)
	for trainLoader.Scan() {
		torch.GC()
		trainLoader.Data()
		torch.FinishGC()
	}
}

func main() {
	trainLoop()
}
