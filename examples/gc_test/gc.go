package main

import (
	"fmt"
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
)

func f() {
	fmt.Println("Entering f()")
	b := torch.RandN(4, 1, true)
	fmt.Println(b)
	fmt.Println("Exiting f()")
}

func main() {
	f()
	fmt.Println("Calling runtime.GC()")
	runtime.GC()
}
