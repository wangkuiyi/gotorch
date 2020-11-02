package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	torch "github.com/wangkuiyi/gotorch"
)

var masterAddr = flag.String("masterAddr", "127.0.0.1", "The address of master node(rank 0)")
var masterPort = flag.Int("masterPort", 11111, "The port of master node")
var rank = flag.Int("rank", 0, "The rank of the current process")
var size = flag.Int("size", 0, "The size of the processes")

func main() {
	flag.Parse()

	f, err := os.OpenFile(fmt.Sprintf("%d.log", *rank), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)

	a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	ts := torch.NewTCPStore(*masterAddr, int64(*masterPort), int64(*size), *rank == 0)
	pg := torch.NewProcessGroupGloo(ts, int64(*rank), int64(*size))

	pg.AllReduce([]torch.Tensor{a})
	log.Println(a)
}
