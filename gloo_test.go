package gotorch

import (
	"io/ioutil"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGlooAllReduce(t *testing.T) {
	f, _ := ioutil.TempFile("", "sample")
	defer os.Remove(f.Name())

	a := NewTensor([][]float32{{1, 2}, {3, 4}})
	b := NewTensor([][]float32{{4, 3}, {2, 1}})

	ts := []Tensor{a, b}
	wg := sync.WaitGroup{}
	wg.Add(2)

	for i := 0; i < 2; i++ {
		go func(rank int64, a Tensor) {
			defer wg.Done()
			fs := NewFileStore(f.Name(), 2)
			defer fs.Close()
			pg := NewProcessGroupGloo(fs, rank, 2)
			defer pg.Close()
			pg.AllReduce([]Tensor{a})
		}(int64(i), ts[i])
	}

	wg.Wait()

	assert.Equal(t, " 5  5\n 5  5\n[ CPUFloatType{2,2} ]", a.String())
	assert.Equal(t, " 5  5\n 5  5\n[ CPUFloatType{2,2} ]", b.String())
}

func TestGlooAllReduceWithTCPStore(t *testing.T) {
	f, _ := ioutil.TempFile("", "sample")
	defer os.Remove(f.Name())

	a1 := NewTensor([][]float32{{1, 2}})
	a2 := NewTensor([][]float32{{1, 3}})
	a := []Tensor{a1, a2}

	b1 := NewTensor([][]float32{{4, 3}})
	b2 := NewTensor([][]float32{{1, 1}})
	b := []Tensor{b1, b2}

	ts := [][]Tensor{a, b}

	wg := sync.WaitGroup{}
	wg.Add(2)

	for i := 0; i < 2; i++ {
		go func(rank int64, a []Tensor) {
			defer wg.Done()
			fs := NewFileStore(f.Name(), 2)
			defer fs.Close()
			pg := NewProcessGroupGloo(fs, rank, 2)
			defer pg.Close()
			pg.AllReduce(a)
		}(int64(i), ts[i])
	}

	wg.Wait()

	assert.Equal(t, " 7  9\n[ CPUFloatType{1,2} ]", a1.String())
	assert.Equal(t, " 7  9\n[ CPUFloatType{1,2} ]", b1.String())
	assert.Equal(t, " 7  9\n[ CPUFloatType{1,2} ]", a2.String())
	assert.Equal(t, " 7  9\n[ CPUFloatType{1,2} ]", b2.String())

}

func TestGlooAllReduceCoalesced(t *testing.T) {
	f, _ := ioutil.TempFile("", "sample")
	defer os.Remove(f.Name())

	a1 := NewTensor([][]float32{{1, 2}})
	a2 := NewTensor([][]float32{{1, 3}})
	a := []Tensor{a1, a2}

	b1 := NewTensor([][]float32{{4, 3}})
	b2 := NewTensor([][]float32{{1, 1}})
	b := []Tensor{b1, b2}

	ts := [][]Tensor{a, b}

	wg := sync.WaitGroup{}
	wg.Add(2)

	for i := 0; i < 2; i++ {
		go func(rank int64, a []Tensor) {
			defer wg.Done()
			fs := NewFileStore(f.Name(), 2)
			defer fs.Close()
			pg := NewProcessGroupGloo(fs, rank, 2)
			defer pg.Close()
			pg.AllReduceCoalesced(a)
		}(int64(i), ts[i])
	}

	wg.Wait()

	assert.Equal(t, " 5  5\n[ CPUFloatType{1,2} ]", a1.String())
	assert.Equal(t, " 5  5\n[ CPUFloatType{1,2} ]", b1.String())
	assert.Equal(t, " 2  4\n[ CPUFloatType{1,2} ]", a2.String())
	assert.Equal(t, " 2  4\n[ CPUFloatType{1,2} ]", b2.String())
}

func TestGlooBroadcast(t *testing.T) {
	f, _ := ioutil.TempFile("", "sample")
	defer os.Remove(f.Name())

	a1 := NewTensor([][]float32{{1, 2}})
	a2 := NewTensor([][]float32{{1, 3}})
	a := []Tensor{a1, a2}

	b1 := NewTensor([][]float32{{4, 3}})
	b2 := NewTensor([][]float32{{1, 1}})
	b := []Tensor{b1, b2}

	ts := [][]Tensor{a, b}

	wg := sync.WaitGroup{}
	wg.Add(2)

	for i := 0; i < 2; i++ {
		go func(rank int64, a []Tensor) {
			defer wg.Done()
			fs := NewFileStore(f.Name(), 2)
			defer fs.Close()
			pg := NewProcessGroupGloo(fs, rank, 2)
			defer pg.Close()
			pg.Broadcast(a)
		}(int64(i), ts[i])
	}

	wg.Wait()

	assert.Equal(t, " 1  2\n[ CPUFloatType{1,2} ]", a1.String())
	assert.Equal(t, " 1  2\n[ CPUFloatType{1,2} ]", b1.String())
	assert.Equal(t, " 1  2\n[ CPUFloatType{1,2} ]", a2.String())
	assert.Equal(t, " 1  2\n[ CPUFloatType{1,2} ]", b2.String())
}
