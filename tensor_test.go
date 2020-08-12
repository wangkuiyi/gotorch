package gotorch_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
)

func ExampleTensor() {
	t := torch.RandN([]int64{10, 100}, false)
	t.Close()
	t.Close()
	// Output:
}

func TestLogSoftmax(t *testing.T) {
	a := assert.New(t)
	x := torch.RandN([]int64{1, 6}, false)
	out := x.LogSoftmax(1)
	// TODO(yancey1989): convert torchTensro as Go slice, that we can
	// check the value.
	a.NotNil(out.T)
}

func TestDetach(t *testing.T) {
	torch.GC()
	x := torch.RandN([]int64{1, 6}, true)
	y := x.Detach()
	assert.NotNil(t, y.T)
	torch.FinishGC()
}
