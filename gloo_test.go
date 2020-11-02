package gotorch

import (
	"io/ioutil"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func allreduce(rank, size int64, a Tensor, f *os.File, wg *sync.WaitGroup) {
	defer wg.Done()

	fs := NewFileStore(f.Name(), size)
	pg := NewProcessGroupGloo(fs, rank, size)
	pg.AllReduce([]Tensor{a})
}

func TestGlooAllReduce(t *testing.T) {
	f, _ := ioutil.TempFile("", "sample")
	defer os.Remove(f.Name())

	a := NewTensor([][]float32{{1, 2}, {3, 4}})
	b := NewTensor([][]float32{{4, 3}, {2, 1}})
	wg := sync.WaitGroup{}
	wg.Add(2)

	go allreduce(0, 2, a, f, &wg)
	go allreduce(1, 2, b, f, &wg)

	wg.Wait()

	assert.Equal(t, " 5  5\n 5  5\n[ CPUFloatType{2,2} ]", a.String())
	assert.Equal(t, " 5  5\n 5  5\n[ CPUFloatType{2,2} ]", b.String())
}
