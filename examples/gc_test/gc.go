package main

import (
	"fmt"
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
)

func f() {
	b := torch.RandN(4, 1, true)
	runtime.SetFinalizer(&b, func(f *torch.Tensor) {
		f.Close()
		fmt.Println("Closed tensor")
	})
}

func main() {
	f()
	runtime.GC()
}
