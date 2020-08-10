package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func ExampleTensor() {
	t := torch.RandN([]int{10, 100}, false)
	t.Close()
	t.Close()
	// Output:
}

func TestTranspose2d(t *testing.T) {
	input := torch.RandN([]int{1, 1, 1}, false)
	weight := torch.RandN([]int{1, 3, 3}, false)
	var bias torch.Tensor
	stride := []int{1}
	padding := []int{0}
	outputPadding := []int{0}
	groups := 1
	dilation := []int{1}
	out := torch.ConvTranspose2d(input, weight, bias,
		stride, padding, outputPadding, groups, dilation)
	a := assert.New(t)
	a.NotNil(out.T)
}

func TestSoftmax(t *testing.T) {
	a := assert.new(t)
	x := torch.RandN([]int{1, 6}, false)
	out := x.Softmax(1)
	// TODO(yancey1989): convert torchTensro as Go slice, that we can
	// check the value.
	a.NotNil(out.T)
}
