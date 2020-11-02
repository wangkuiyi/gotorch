package main

import (
	"flag"
	"fmt"
	"os/exec"
	"strings"
)

var numNodes = flag.Int("numNodes", 1, "The number of nodes for distributed training")
var nodeRank = flag.Int("nodeRank", 0, "The rank of the node")
var nprocPerNode = flag.Int("nprocPerNode", 1, "The number of processes on each node")
var masterAddr = flag.String("masterAddr", "127.0.0.1", "The address of master node(rank 0)")
var masterPort = flag.Int("masterPort", 11111, "The port of master node")
var sharedFile = flag.String("sharedFile", "", "The shared file which could be access by all processes")
var trainingCmd = flag.String("trainingCmd", "", "The training command")

func main() {
	flag.Parse()

	commands := []string{}
	size := (*numNodes) * (*nprocPerNode)
	for i := 0; i < *nprocPerNode; i++ {
		rank := (*nprocPerNode)*(*nodeRank) + i
		cmd := fmt.Sprintf("%s -rank=%d -size=%d", *trainingCmd, rank, size)
		if *masterAddr != "" {
			cmd = fmt.Sprintf("%s -masterAddr=%s -masterPort=%d", cmd, *masterAddr, *masterPort)
		} else if *sharedFile != "" {
			cmd = fmt.Sprintf("%s -sharedFile=%s", cmd, *sharedFile)
		} else {
			panic("Must set value for masterAddr or sharedFile")
		}
		commands = append(commands, cmd)
	}

	for _, cmd := range commands {
		args := strings.Fields(cmd)
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Start()
	}
}
