package main

import (
	torch "github.com/wangkuiyi/gotorch"
)

// Net struct
type Net struct {
	l1, l2 *torch.Linear
}

// NewNet creats a net instance
func NewNet(m *torch.Model) *Net {
	return &Net{
		l1: torch.NewLinear(m, 100, 200, false),
		l2: torch.NewLinear(m, 200, 10, false),
	}
}

// Forward executes forward calculation
func (n *Net) Forward(x torch.Tensor) torch.Tensor {
	x = n.l1.Forward(x)
	x = n.l2.Forward(x)
	return x
}

func main() {
	model := torch.NewModel()
	net := NewNet(model)

	opt := torch.NewSGDOpt(0.1, 0, 0, 0, false)
	opt.AddParameters(model.Parameters)

	for i := 0; i < 1000; i++ {
		data := torch.RandN(32, 100, false)
		pre := net.Forward(data)
		loss := torch.Sum(pre)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()

		loss.Close()
		model.CloseVariables()
		data.Close()
	}

	opt.Close()
	model.CloseParameters()
}
