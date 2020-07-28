package main

import (
	"fmt"

	"github.com/wangkuiyi/gotorch/torch"
	at "github.com/wangkuiyi/gotorch/aten"
)

func main() {
	a := torch.RandN(3, 4, true)
	fmt.Println(a)

	b := torch.RandN(4, 1, true)
	fmt.Println(b)

	c := at.MM(a, b)
	fmt.Println(c)

	d := at.Sum(c)
	fmt.Println(d)

	d.Backward()

	fmt.Println(a.Grad())
	fmt.Println(b.Grad())
}
