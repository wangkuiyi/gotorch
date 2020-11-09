package recordio

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/recordio"
)

func TestReader(t *testing.T) {
	testData := &ImageRecord{
		Image: []byte{1, 2},
		Label: 0,
	}
	b, _ := testData.Encode()

	f1, _ := ioutil.TempFile("", "sample1")
	defer os.Remove(f1.Name())
	w1 := recordio.NewWriter(f1, -1, -1)
	i1 := 6
	for i := 0; i < i1; i++ {
		w1.Write(b)
	}
	w1.Close()

	f2, _ := ioutil.TempFile("", "sample2")
	defer os.Remove(f2.Name())
	w2 := recordio.NewWriter(f1, -1, -1)
	i2 := 18
	for i := 0; i < i2; i++ {
		w2.Write(b)
	}
	w2.Close()

	files := []string{f1.Name(), f2.Name()}
	r, e := NewReader(files)
	defer r.Close()
	assert.Nil(t, e)

	i := 0
	for {
		ir, _ := r.Next()
		if ir == nil {
			break
		}
		i++
	}
	assert.Equal(t, i1+i2, i)
}
