package main

import (
	"fmt"
	"image"
	"os"

	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func main() {
	filePath := "188242.jpg"
	f, _ := os.Open(filePath)
	image, _, _ := image.Decode(f)
	out := transforms.ToTensor().Run(image)
	fmt.Println(out.Index([]int64{2, 217, 177}))
}
