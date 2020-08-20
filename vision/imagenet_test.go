package vision

import "testing"

var imageNetRoot = "../testdata/imagenet/train"

func TestImageNet(t *testing.T) {
	imageNet := NewImageNet(imageNetRoot, 32, []Transform{})
	for imageNet.Scan() {
		imageNet.Batch()
	}
}
