package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	torch_recordio "github.com/wangkuiyi/gotorch/tool/recordio"
	"github.com/wangkuiyi/recordio"
)

var dataset = flag.String("dataset", "", "path to dataset directory")
var label = flag.String("label", "", "path to label txt file")
var output = flag.String("output", "", "output record files directory")
var shuffle = flag.Bool("shuffle", true, "shuffle dataset")
var recordsPerShard = flag.Int("recordsPerShard", 4096, "maximum number of records per shard file")

func loadImage(fname string, vocab map[string]int) (*torch_recordio.ImageRecord, error) {
	b, err := ioutil.ReadFile(fname)
	if err != nil {
		return nil, err
	}
	classStr := filepath.Base(filepath.Dir(fname))
	ir := &torch_recordio.ImageRecord{
		Image: b,
		Label: vocab[classStr],
	}
	return ir, nil
}

func buildLabelVocabulary(label string) (map[string]int, error) {
	file, err := os.Open(label)
	if err != nil {
		return nil, nil
	}
	defer file.Close()

	vocab := map[string]int{}
	scanner := bufio.NewScanner(file)
	idx := 0
	for scanner.Scan() {
		vocab[strings.TrimSpace(scanner.Text())] = idx
		idx++
	}
	return vocab, nil
}

func create(p string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(p), 0770); err != nil {
		return nil, err
	}
	return os.Create(p)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	flag.Parse()
	vocab, err := buildLabelVocabulary(*label)
	if err != nil {
		panic(err)
	}

	files, err := filepath.Glob(*dataset + "/*/*.*")
	if *shuffle {
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	}

	if err != nil {
		panic(err)
	}

	start := []int{}
	end := []int{}
	for i := 0; i < len(files); i += *recordsPerShard {
		start = append(start, i)
		end = append(end, min(i+*recordsPerShard, len(files)))
	}

	var wg sync.WaitGroup
	wg.Add(len(start))

	for i := 0; i < len(start); i++ {
		go func(i int) {
			s := start[i]
			e := end[i]
			f, _ := create(fmt.Sprintf("%s/data-%05d", *output, i))
			w := recordio.NewWriter(f, -1, -1)
			for row := s; row < e; row++ {
				ir, _ := loadImage(files[row], vocab)
				b, _ := ir.Encode()
				w.Write(b)
			}
			w.Close()
			f.Close()
			wg.Done()
		}(i)
	}

	wg.Wait()
}
