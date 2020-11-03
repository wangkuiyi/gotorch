package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	torch "github.com/wangkuiyi/gotorch"
	F "github.com/wangkuiyi/gotorch/nn/functional"
	"github.com/wangkuiyi/gotorch/vision/models"
)

var masterAddr = flag.String("masterAddr", "127.0.0.1", "The address of master node(rank 0)")
var masterPort = flag.Int("masterPort", 11111, "The port of master node")
var rank = flag.Int("rank", 0, "The rank of the current process")
var size = flag.Int("size", 1, "The size of the processes")

func getGrads(params []torch.Tensor) (grads []torch.Tensor) {
	for _, p := range params {
		grads = append(grads, p.Grad())
	}
	return
}

func main() {
	flag.Parse()

	f, err := os.OpenFile(fmt.Sprintf("%d.log", *rank), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)

	ts := torch.NewTCPStore(*masterAddr, int64(*masterPort), int64(*size), *rank == 0)
	defer ts.Close()
	pg := torch.NewProcessGroupGloo(ts, int64(*rank), int64(*size))
	defer pg.Close()

	net := models.MLP()
	opt := torch.SGD(0.01, 0.5, 0, 0, false)
	params := net.Parameters()
	opt.AddParameters(params)

	log.Println(params[0].Index(0).Item())
	for _, p := range params {
		pg.Broadcast([]torch.Tensor{p})
	}
	log.Println(params[0].Index(0).Item())

	for i := 0; i < 10; i++ {
		data := torch.Rand([]int64{16, 28, 28}, false)
		label := torch.Ones([]int64{16}, false).CastTo(torch.Long)

		opt.ZeroGrad()
		pred := net.Forward(data)
		loss := F.NllLoss(pred, label, torch.Tensor{}, -100, "mean")
		loss.Backward()

		grads := getGrads(params)

		log.Println(grads[0].Index(0).Item())
		pg.AllReduceCoalesced(grads)
		log.Println(grads[0].Index(0).Item())

		opt.Step()
	}
}
