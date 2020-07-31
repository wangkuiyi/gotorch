package main

import (
	"fmt"
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
)

func f() {
	b := torch.RandN(4, 1, true)
	fmt.Println(b)
}

func main() {
	f()
	runtime.GC()
}
