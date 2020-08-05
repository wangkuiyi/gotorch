package gotorch

import "testing"

func TestOptim(t *testing.T) {
	b := RandN(4, 1, true)
	opt := SGD(0.1, 0, 0, 0, false)
	opt.AddParameters([]Tensor{b})

	a := RandN(3, 4, false)
	c := MM(a, b)
	d := Sum(c)

	opt.ZeroGrad()
	d.Backward()
	opt.Step()
}
