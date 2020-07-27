package main

import (
	"fmt"
	"github.com/wangkuiyi/gotorch/at"
)

func main() {
	a := at.RandN(3, 4, true)
	fmt.Println(a)

	b := at.RandN(4, 1, true)
	fmt.Println(b)

	c := at.MM(a, b)
	fmt.Println(c)

	d := at.Sum(c)
	fmt.Println(d)

	d.Backward()

	fmt.Println(a.Grad())
	fmt.Println(b.Grad())
}
