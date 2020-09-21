package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/wangkuiyi/gotorch/vision/imageloader"
)

var data = flag.String("data", "", "path to dataset")
var label = flag.String("label", "", "path to save labels")

// SaveLabel saves label to a gob file
func SaveLabel(label map[string]int, fileName string) error {
	f, e := os.Create(fileName)
	if e != nil {
		return fmt.Errorf("Fail to create file to save labels: %v", e)
	}
	defer f.Close()

	if e := gob.NewEncoder(f).Encode(label); e != nil {
		return e
	}
	return nil
}

func main() {
	flag.Parse()

	labels, err := imageloader.BuildLabelVocabularyFromTgz(*data)
	if err != nil {
		log.Fatal("Fail to build labels")
	}

	err = SaveLabel(labels, *label)
	if err != nil {
		log.Fatal("Fail to save labels")
	}
}
