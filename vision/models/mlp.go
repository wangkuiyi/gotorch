package models

import (
	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
)

// MLPModule represent a multilayer perceptron network
type MLPModule struct {
	nn.Module
	FC1, FC2, FC3 *nn.LinearModule
}

// Forward runs the forward pass
func (n *MLPModule) Forward(x torch.Tensor) torch.Tensor {
	x = torch.View(x, -1, 28*28)
	x = n.FC1.Forward(x)
	x = torch.Tanh(x)
	x = n.FC2.Forward(x)
	x = torch.Tanh(x)
	x = n.FC3.Forward(x)
	return x.LogSoftmax(1)
}

// MLP returns MLPModule
func MLP() *MLPModule {
	r := &MLPModule{
		FC1: nn.Linear(28*28, 512, true),
		FC2: nn.Linear(512, 512, true),
		FC3: nn.Linear(512, 10, true)}
	r.Init(r)
	return r
}
